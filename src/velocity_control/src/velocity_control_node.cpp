#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <control_msgs/action/gripper_command.hpp>
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <Eigen/Dense>
#include <memory>
#include <chrono>
#include <deque>
#include <cmath>
#include <fstream>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std::chrono_literals;
using GripperCommand = control_msgs::action::GripperCommand;

class VelocityControlNode : public rclcpp::Node
{
public:
    VelocityControlNode() : Node("velocity_control_node")
    {
        // Declare parameters
        this->declare_parameter("robot_urdf_path", "/home/ming/xrrobotics_new/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/GEN3-7DOF.urdf");
        this->declare_parameter("control_frequency", 100.0);
        this->declare_parameter("gripper_control_frequency", 10.0);  // Lower frequency for gripper
        this->declare_parameter("base_link", "base_link");
        this->declare_parameter("end_effector_link", "bracelet_link");
        
        // Velocity control parameters
        this->declare_parameter("position_gain", 2.0);  // P gain for position error
        this->declare_parameter("orientation_gain", 2.0);  // P gain for orientation error
        this->declare_parameter("max_linear_velocity", 0.5);  // m/s
        this->declare_parameter("max_angular_velocity", 1.0);  // rad/s
        this->declare_parameter("max_linear_acceleration", 10.0);  // m/s^2
        this->declare_parameter("max_angular_acceleration", 5.0);  // rad/s^2
        this->declare_parameter("velocity_filter_alpha", 0.5);  // Low-pass filter coefficient
        this->declare_parameter("deadzone", 0.002);  // 2mm deadzone for position
        this->declare_parameter("angular_deadzone", 0.01);  // ~0.57 degrees deadzone for orientation
        this->declare_parameter("publish_angular_in_degrees", true);

        
        // Get parameters
        urdf_path_ = this->get_parameter("robot_urdf_path").as_string();
        control_frequency_ = this->get_parameter("control_frequency").as_double();
        gripper_control_frequency_ = this->get_parameter("gripper_control_frequency").as_double();
        base_link_ = this->get_parameter("base_link").as_string();
        end_effector_link_ = this->get_parameter("end_effector_link").as_string();
        
        position_gain_ = this->get_parameter("position_gain").as_double();
        orientation_gain_ = this->get_parameter("orientation_gain").as_double();
        max_linear_vel_ = this->get_parameter("max_linear_velocity").as_double();
        max_angular_vel_ = this->get_parameter("max_angular_velocity").as_double();
        max_linear_acc_ = this->get_parameter("max_linear_acceleration").as_double();
        max_angular_acc_ = this->get_parameter("max_angular_acceleration").as_double();
        filter_alpha_ = this->get_parameter("velocity_filter_alpha").as_double();
        position_deadzone_ = this->get_parameter("deadzone").as_double();
        angular_deadzone_ = this->get_parameter("angular_deadzone").as_double();
        publish_ang_in_degrees_ = this->get_parameter("publish_angular_in_degrees").as_bool();

        R_ctrl_from_rviz_ << 0, 1, 0,
                             0, 0, 1,
                             1, 0, 0;
        dt_ = 1.0 / control_frequency_;
        
        // Initialize FK solver
        if (!initializeFKSolver()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize FK solver");
            throw std::runtime_error("FK solver initialization failed");
        }
        
        // Initialize velocity states
        filtered_linear_vel_.setZero();
        filtered_angular_vel_.setZero();
        last_linear_vel_.setZero();
        last_angular_vel_.setZero();
        
        // Create subscribers
        target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/cartesian_target", 10,
            std::bind(&VelocityControlNode::targetPoseCallback, this, std::placeholders::_1));
        
        gripper_cmd_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/gripper_command", 10,
            std::bind(&VelocityControlNode::gripperCommandCallback, this, std::placeholders::_1));
        
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&VelocityControlNode::jointStateCallback, this, std::placeholders::_1));
        
        // Create publishers
        twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/twist_controller/commands", 10);
        
        // Create action client for gripper control
        gripper_action_client_ = rclcpp_action::create_client<GripperCommand>(
            this, "/robotiq_gripper_controller/gripper_cmd");
        
        // Wait for action server
        if (!gripper_action_client_->wait_for_action_server(5s)) {
            RCLCPP_WARN(this->get_logger(), 
                       "Gripper action server not available after waiting 5 seconds. "
                       "Gripper control will not work.");
        } else {
            RCLCPP_INFO(this->get_logger(), "Gripper action server is available");
        }
        
                // 配置 action 回调（非阻塞）
        send_goal_options_.goal_response_callback =
        [this](rclcpp_action::ClientGoalHandle<GripperCommand>::SharedPtr handle)
        {
            if (!handle) {
            RCLCPP_WARN(this->get_logger(), "Gripper goal rejected");
            } else {
            this->current_gripper_goal_ = handle;  // 存句柄，后续可取消
            RCLCPP_DEBUG(this->get_logger(), "Gripper goal accepted");
            }
        };

        send_goal_options_.feedback_callback =
        [this](rclcpp_action::ClientGoalHandle<GripperCommand>::SharedPtr /*handle*/,
                const std::shared_ptr<const GripperCommand::Feedback> feedback)
        {
            // 可选：打印/使用反馈
            // RCLCPP_DEBUG(this->get_logger(), "Gripper pos=%.3f, effort=%.3f",
            //              feedback->position, feedback->effort);
        };

        send_goal_options_.result_callback =
        [this](const rclcpp_action::ClientGoalHandle<GripperCommand>::WrappedResult &result)
        {
            if (result.code != rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_WARN(this->get_logger(), "Gripper goal finished with code %d", (int)result.code);
            } else {
            RCLCPP_DEBUG(this->get_logger(), "Gripper goal succeeded");
            }
        };

        // Create service client for controller switching
        controller_switch_client_ = this->create_client<controller_manager_msgs::srv::SwitchController>(
            "/controller_manager/switch_controller");
        
        // Wait for controller manager service
        while (!controller_switch_client_->wait_for_service(1s)) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for controller_manager service");
                return;
            }
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                "Waiting for controller_manager service...");
        }
        
        // Switch to twist controller
        if (!switchToTwistController()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to switch to twist controller");
            throw std::runtime_error("Controller switch failed");
        }
        
        // Create control timer for velocity control
        auto period = std::chrono::duration<double>(dt_);
        control_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&VelocityControlNode::controlLoop, this));
        
        // Create timer for gripper control (lower frequency)
        auto gripper_period = std::chrono::duration<double>(1.0 / gripper_control_frequency_);
        gripper_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(gripper_period),
            std::bind(&VelocityControlNode::gripperControlLoop, this));
        
        RCLCPP_INFO(this->get_logger(), "Velocity Control Node initialized");
        RCLCPP_INFO(this->get_logger(), "Position gain: %.2f, Orientation gain: %.2f", 
                   position_gain_, orientation_gain_);
        RCLCPP_INFO(this->get_logger(), "Max linear vel: %.2f m/s, Max angular vel: %.2f rad/s", 
                   max_linear_vel_, max_angular_vel_);
        RCLCPP_INFO(this->get_logger(), "Control frequency: %.1f Hz, Gripper frequency: %.1f Hz", 
                   control_frequency_, gripper_control_frequency_);
    }
    
    ~VelocityControlNode()
    {
        // Try to switch back to joint trajectory controller on shutdown
        switchToJointTrajectoryController();
    }
    
private:
    // FK solver for current pose computation
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    KDL::Chain kdl_chain_;
    KDL::JntArray current_joint_positions_;
    KDL::Frame current_ee_frame_;
    std::vector<std::string> joint_names_;
    std::map<std::string, int> joint_name_to_index_;
    int num_joints_;
    
    // Parameters
    std::string urdf_path_, base_link_, end_effector_link_;
    double control_frequency_, gripper_control_frequency_, dt_;
    double position_gain_, orientation_gain_;
    double max_linear_vel_, max_angular_vel_;
    double max_linear_acc_, max_angular_acc_;
    double filter_alpha_;
    double position_deadzone_, angular_deadzone_;
    
    // Control states
    geometry_msgs::msg::PoseStamped target_pose_;
    bool target_received_ = false;
    bool joints_initialized_ = false;
    bool publish_ang_in_degrees_ = true;

    // Velocity states
    Eigen::Vector3d filtered_linear_vel_;
    Eigen::Vector3d filtered_angular_vel_;
    Eigen::Vector3d last_linear_vel_;
    Eigen::Vector3d last_angular_vel_;
    Eigen::Matrix3d R_ctrl_from_rviz_;
    // Gripper state
    double gripper_command_ = 0.0;
    double last_gripper_command_ = -1.0;  // Initialize to invalid value to force first send
    rclcpp_action::Client<GripperCommand>::SendGoalOptions send_goal_options_;
    rclcpp_action::Client<GripperCommand>::GoalHandle::SharedPtr current_gripper_goal_;
    
    // ROS interfaces
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr gripper_cmd_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
    rclcpp_action::Client<GripperCommand>::SharedPtr gripper_action_client_;
    rclcpp::Client<controller_manager_msgs::srv::SwitchController>::SharedPtr controller_switch_client_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::TimerBase::SharedPtr gripper_timer_;
    
    bool initializeFKSolver()
    {
        // Read URDF file
        std::ifstream urdf_file(urdf_path_);
        if (!urdf_file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open URDF: %s", urdf_path_.c_str());
            return false;
        }
        
        std::string urdf_string((std::istreambuf_iterator<char>(urdf_file)),
                                std::istreambuf_iterator<char>());
        
        // Parse URDF to KDL tree
        KDL::Tree kdl_tree;
        if (!kdl_parser::treeFromString(urdf_string, kdl_tree)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF");
            return false;
        }
        
        // Extract chain
        if (!kdl_tree.getChain(base_link_, end_effector_link_, kdl_chain_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to extract chain");
            return false;
        }
        
        num_joints_ = kdl_chain_.getNrOfJoints();
        current_joint_positions_.resize(num_joints_);
        
        // Initialize FK solver
        fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(kdl_chain_);
        
        // Get joint names
        for (unsigned int i = 0; i < kdl_chain_.getNrOfSegments(); ++i) {
            const KDL::Segment& segment = kdl_chain_.getSegment(i);
            const KDL::Joint& joint = segment.getJoint();
            if (joint.getType() != KDL::Joint::None) {
                joint_names_.push_back(joint.getName());
                joint_name_to_index_[joint.getName()] = joint_names_.size() - 1;
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "FK solver initialized with %d joints", num_joints_);
        return true;
    }
    
    bool switchToTwistController()
    {
        auto request = std::make_shared<controller_manager_msgs::srv::SwitchController::Request>();
        request->activate_controllers.push_back("twist_controller");
        request->deactivate_controllers.push_back("joint_trajectory_controller");
        request->strictness = controller_manager_msgs::srv::SwitchController::Request::BEST_EFFORT;
        request->activate_asap = true;
        
        RCLCPP_INFO(this->get_logger(), "Switching to twist_controller...");
        
        auto future = controller_switch_client_->async_send_request(request);
        
        // Wait for the result
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = future.get();
            if (response->ok) {
                RCLCPP_INFO(this->get_logger(), "Successfully switched to twist_controller");
                return true;
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to switch to twist_controller");
                return false;
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Service call to switch controller failed");
            return false;
        }
    }
    
    bool switchToJointTrajectoryController()
    {
        auto request = std::make_shared<controller_manager_msgs::srv::SwitchController::Request>();
        request->activate_controllers.push_back("joint_trajectory_controller");
        request->deactivate_controllers.push_back("twist_controller");
        request->strictness = controller_manager_msgs::srv::SwitchController::Request::BEST_EFFORT;
        request->activate_asap = true;
        
        RCLCPP_INFO(this->get_logger(), "Switching back to joint_trajectory_controller...");
        
        auto future = controller_switch_client_->async_send_request(request);
        
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS)
        {
            auto response = future.get();
            if (response->ok) {
                RCLCPP_INFO(this->get_logger(), "Successfully switched to joint_trajectory_controller");
                return true;
            }
        }
        return false;
    }
    
    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        target_pose_ = *msg;
        target_received_ = true;
    }
    
    void gripperCommandCallback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        gripper_command_ = msg->data;
    }
    
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // Update current joint positions
        for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
            auto it = joint_name_to_index_.find(msg->name[i]);
            if (it != joint_name_to_index_.end()) {
                current_joint_positions_(it->second) = msg->position[i];
            }
        }
        
        if (!joints_initialized_) {
            joints_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Joint states initialized");
        }
        
        // Compute current end-effector pose using FK
        if (joints_initialized_) {
            fk_solver_->JntToCart(current_joint_positions_, current_ee_frame_);
        }
    }
    
    geometry_msgs::msg::PoseStamped kdlFrameToPoseMsg(const KDL::Frame& frame)
    {
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = this->now();
        pose_msg.header.frame_id = base_link_;
        
        // Position
        pose_msg.pose.position.x = frame.p.x();
        pose_msg.pose.position.y = frame.p.y();
        pose_msg.pose.position.z = frame.p.z();
        
        // Orientation
        double x, y, z, w;
        frame.M.GetQuaternion(x, y, z, w);
        pose_msg.pose.orientation.x = x;
        pose_msg.pose.orientation.y = y;
        pose_msg.pose.orientation.z = z;
        pose_msg.pose.orientation.w = w;
        
        return pose_msg;
    }
    
    Eigen::Vector3d computeOrientationError(const Eigen::Quaterniond& current_quat,
                                           const Eigen::Quaterniond& target_quat)
    {
        // Compute quaternion error
        Eigen::Quaterniond quat_error = target_quat * current_quat.conjugate();
        
        // Convert to angle-axis
        Eigen::AngleAxisd angle_axis(quat_error);
        
        // Return axis-angle representation
        return angle_axis.angle() * angle_axis.axis();
    }
    
    Eigen::Vector3d applyDeadzone(const Eigen::Vector3d& value, double deadzone)
    {
        Eigen::Vector3d result;
        for (int i = 0; i < 3; ++i) {
            if (std::abs(value[i]) < deadzone) {
                result[i] = 0.0;
            } else {
                result[i] = value[i];
            }
        }
        return result;
    }
    
    Eigen::Vector3d limitVector(const Eigen::Vector3d& vec, double max_magnitude)
    {
        double magnitude = vec.norm();
        if (magnitude > max_magnitude && magnitude > 1e-6) {
            return vec * (max_magnitude / magnitude);
        }
        return vec;
    }
    
    Eigen::Vector3d applyAccelerationLimit(const Eigen::Vector3d& desired_vel,
                                          const Eigen::Vector3d& last_vel,
                                          double max_acc)
    {
        Eigen::Vector3d vel_diff = desired_vel - last_vel;
        double max_change = max_acc * dt_;
        
        // Limit the velocity change
        vel_diff = limitVector(vel_diff, max_change);
        
        return last_vel + vel_diff;
    }


    static inline Eigen::Vector3d remap_rviz_to_controller(const Eigen::Vector3d& v_rviz)
    {
        // RViz/base_link: [x, y, z]  ->  Controller: [y, z, x]
        return Eigen::Vector3d(v_rviz.y(), v_rviz.z(), v_rviz.x());
    }
    
    void controlLoop()
    {
        if (!target_received_ || !joints_initialized_) {
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Waiting for target pose and joint states...");
            return;
        }
        
        // Get current pose from FK
        auto current_pose = kdlFrameToPoseMsg(current_ee_frame_);
        
        // Compute position error
        Eigen::Vector3d position_error;
        position_error << target_pose_.pose.position.x - current_pose.pose.position.x,
                         target_pose_.pose.position.y - current_pose.pose.position.y,
                         target_pose_.pose.position.z - current_pose.pose.position.z;
        
        // Apply deadzone to position error
        position_error = applyDeadzone(position_error, position_deadzone_);
        
        // Compute orientation error
        Eigen::Quaterniond current_quat(
            current_pose.pose.orientation.w,
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z
        );
        
        Eigen::Quaterniond target_quat(
            target_pose_.pose.orientation.w,
            target_pose_.pose.orientation.x,
            target_pose_.pose.orientation.y,
            target_pose_.pose.orientation.z
        );
        
        Eigen::Vector3d orientation_error = computeOrientationError(current_quat, target_quat);
        
        // Apply deadzone to orientation error
        orientation_error = applyDeadzone(orientation_error, angular_deadzone_);
        
        // Compute desired velocities (proportional control)
        Eigen::Vector3d desired_linear_vel = position_gain_ * position_error;
        Eigen::Vector3d desired_angular_vel = orientation_gain_ * orientation_error;
        
        // Limit velocities
        desired_linear_vel = limitVector(desired_linear_vel, max_linear_vel_);
        desired_angular_vel = limitVector(desired_angular_vel, max_angular_vel_);
        
        // Apply acceleration limits
        desired_linear_vel = applyAccelerationLimit(desired_linear_vel, last_linear_vel_, max_linear_acc_);
        desired_angular_vel = applyAccelerationLimit(desired_angular_vel, last_angular_vel_, max_angular_acc_);
        
        // Apply low-pass filter
        filtered_linear_vel_ = filter_alpha_ * desired_linear_vel + 
                              (1.0 - filter_alpha_) * filtered_linear_vel_;
        filtered_angular_vel_ = filter_alpha_ * desired_angular_vel + 
                               (1.0 - filter_alpha_) * filtered_angular_vel_;
        
                // 低通滤波后的速度（仍在 RViz/base_link 轴向）
        Eigen::Vector3d lin_rviz = filtered_linear_vel_;
        Eigen::Vector3d ang_rviz = filtered_angular_vel_;

        // 用 3x3 置换矩阵映射到控制器坐标
        Eigen::Vector3d lin_ctrl = R_ctrl_from_rviz_ * lin_rviz;
        Eigen::Vector3d ang_ctrl = R_ctrl_from_rviz_ * ang_rviz;

        Eigen::Vector3d ang_ctrl_out = ang_ctrl;  // 默认按 rad/s
        if (publish_ang_in_degrees_) {
            constexpr double RAD2DEG = 180.0 / M_PI;
            ang_ctrl_out *= RAD2DEG;  // 转成 deg/s
        }

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "CTRL Linear vel: [%.3f, %.3f, %.3f] m/s, Angular vel: [%.3f, %.3f, %.3f] %s",
            lin_ctrl.x(), lin_ctrl.y(), lin_ctrl.z(),
            ang_ctrl_out.x(), ang_ctrl_out.y(), ang_ctrl_out.z(),
            publish_ang_in_degrees_ ? "deg/s" : "rad/s");

        // Update last velocities
        
        last_linear_vel_ = filtered_linear_vel_;
        last_angular_vel_ = filtered_angular_vel_;
        
        // Publish twist command
        geometry_msgs::msg::Twist twist_msg;
        twist_msg.linear.x  = lin_ctrl.x();
        twist_msg.linear.y  = lin_ctrl.y();
        twist_msg.linear.z  = lin_ctrl.z();
        twist_msg.angular.x = ang_ctrl_out.x();
        twist_msg.angular.y = ang_ctrl_out.y();
        twist_msg.angular.z = ang_ctrl_out.z();
        
        twist_pub_->publish(twist_msg);
        
        // Debug output
        static int counter = 0;
        if (++counter % static_cast<int>(control_frequency_) == 0) {  // Once per second
            RCLCPP_DEBUG(this->get_logger(), 
                        "Linear vel: [%.3f, %.3f, %.3f] m/s, Angular vel: [%.3f, %.3f, %.3f] rad/s",
                        filtered_linear_vel_.x(), filtered_linear_vel_.y(), filtered_linear_vel_.z(),
                        filtered_angular_vel_.x(), filtered_angular_vel_.y(), filtered_angular_vel_.z());
            RCLCPP_DEBUG(this->get_logger(), 
                        "Position error: [%.3f, %.3f, %.3f] m, Orientation error: [%.3f, %.3f, %.3f] rad",
                        position_error.x(), position_error.y(), position_error.z(),
                        orientation_error.x(), orientation_error.y(), orientation_error.z());
        }
    }
    
    void gripperControlLoop()
    {
    // 变化不大就不发，防抖
    if (std::abs(gripper_command_ - last_gripper_command_) < 0.01) {
        return;
    }

    if (!gripper_action_client_->action_server_is_ready()) {
        RCLCPP_DEBUG(this->get_logger(), "Gripper action server not ready");
        return;
    }

    // 如需百分比→开度映射，请在这里换算；你现在按百分比发就保持不变
    auto goal_msg = GripperCommand::Goal();
    goal_msg.command.position = gripper_command_;   // 你确定服务器按百分比解释即可
    goal_msg.command.max_effort = 20.0;

    // 若上个 goal 仍在执行，发起取消（非阻塞）
    if (current_gripper_goal_ &&
        current_gripper_goal_->get_status() == rclcpp_action::GoalStatus::STATUS_EXECUTING) {
        (void)gripper_action_client_->async_cancel_goal(current_gripper_goal_);
    }

    RCLCPP_DEBUG(this->get_logger(), "Sending gripper command: %.3f", gripper_command_);

    // 直接发即可，future 不用 then，不用 get（避免阻塞）
    (void)gripper_action_client_->async_send_goal(goal_msg, send_goal_options_);

    // 记录这次（用于防抖）
    last_gripper_command_ = gripper_command_;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<VelocityControlNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}