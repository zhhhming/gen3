#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
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

using namespace std::chrono_literals;

class VelocityControlNode : public rclcpp::Node
{
public:
    VelocityControlNode() : Node("velocity_control_node")
    {
        // Declare parameters
        this->declare_parameter("robot_urdf_path", "/home/ming/xrrobotics_new/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/GEN3-7DOF.urdf");
        this->declare_parameter("control_frequency", 100.0);
        this->declare_parameter("base_link", "base_link");
        this->declare_parameter("end_effector_link", "bracelet_link");
        
        // Velocity control parameters
        this->declare_parameter("position_gain", 2.0);  // P gain for position error
        this->declare_parameter("orientation_gain", 1.0);  // P gain for orientation error
        this->declare_parameter("max_linear_velocity", 0.5);  // m/s
        this->declare_parameter("max_angular_velocity", 1.0);  // rad/s
        this->declare_parameter("max_linear_acceleration", 1.0);  // m/s^2
        this->declare_parameter("max_angular_acceleration", 2.0);  // rad/s^2
        this->declare_parameter("velocity_filter_alpha", 0.3);  // Low-pass filter coefficient
        this->declare_parameter("deadzone", 0.002);  // 2mm deadzone for position
        this->declare_parameter("angular_deadzone", 0.01);  // ~0.57 degrees deadzone for orientation
        
        // Get parameters
        urdf_path_ = this->get_parameter("robot_urdf_path").as_string();
        control_frequency_ = this->get_parameter("control_frequency").as_double();
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
        
        current_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/current_ee_pose", 10,
            std::bind(&VelocityControlNode::currentPoseCallback, this, std::placeholders::_1));
        
        gripper_cmd_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/gripper_command", 10,
            std::bind(&VelocityControlNode::gripperCommandCallback, this, std::placeholders::_1));
        
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&VelocityControlNode::jointStateCallback, this, std::placeholders::_1));
        
        // Create publishers
        twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/twist_controller/commands", 10);
        
        gripper_trajectory_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/robotiq_gripper_controller/joint_trajectory", 10);
        
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
        
        // Create control timer
        auto period = std::chrono::duration<double>(dt_);
        control_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&VelocityControlNode::controlLoop, this));
        
        RCLCPP_INFO(this->get_logger(), "Velocity Control Node initialized");
        RCLCPP_INFO(this->get_logger(), "Position gain: %.2f, Orientation gain: %.2f", 
                   position_gain_, orientation_gain_);
        RCLCPP_INFO(this->get_logger(), "Max linear vel: %.2f m/s, Max angular vel: %.2f rad/s", 
                   max_linear_vel_, max_angular_vel_);
        RCLCPP_INFO(this->get_logger(), "Control frequency: %.1f Hz", control_frequency_);
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
    std::vector<std::string> joint_names_;
    std::map<std::string, int> joint_name_to_index_;
    int num_joints_;
    
    // Parameters
    std::string urdf_path_, base_link_, end_effector_link_;
    double control_frequency_, dt_;
    double position_gain_, orientation_gain_;
    double max_linear_vel_, max_angular_vel_;
    double max_linear_acc_, max_angular_acc_;
    double filter_alpha_;
    double position_deadzone_, angular_deadzone_;
    
    // Control states
    geometry_msgs::msg::PoseStamped target_pose_;
    geometry_msgs::msg::PoseStamped current_pose_;
    bool target_received_ = false;
    bool current_received_ = false;
    bool joints_initialized_ = false;
    
    // Velocity states
    Eigen::Vector3d filtered_linear_vel_;
    Eigen::Vector3d filtered_angular_vel_;
    Eigen::Vector3d last_linear_vel_;
    Eigen::Vector3d last_angular_vel_;
    
    // Gripper state
    double gripper_command_ = 0.0;
    
    // ROS interfaces
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr current_pose_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr gripper_cmd_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr gripper_trajectory_pub_;
    rclcpp::Client<controller_manager_msgs::srv::SwitchController>::SharedPtr controller_switch_client_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    
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
    
    void currentPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        current_pose_ = *msg;
        current_received_ = true;
    }
    
    void gripperCommandCallback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        gripper_command_ = msg->data;
        
        // Publish gripper trajectory
        trajectory_msgs::msg::JointTrajectory gripper_traj;
        gripper_traj.header.stamp = this->now();
        gripper_traj.joint_names = {"robotiq_85_left_knuckle_joint"};
        
        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = {gripper_command_};
        point.time_from_start = rclcpp::Duration::from_seconds(0.1);
        
        gripper_traj.points.push_back(point);
        gripper_trajectory_pub_->publish(gripper_traj);
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
    
    void controlLoop()
    {
        if (!target_received_ || !current_received_) {
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Waiting for target and current poses...");
            return;
        }
        
        // Compute position error
        Eigen::Vector3d position_error;
        position_error << target_pose_.pose.position.x - current_pose_.pose.position.x,
                         target_pose_.pose.position.y - current_pose_.pose.position.y,
                         target_pose_.pose.position.z - current_pose_.pose.position.z;
        
        // Apply deadzone to position error
        position_error = applyDeadzone(position_error, position_deadzone_);
        
        // Compute orientation error
        Eigen::Quaterniond current_quat(
            current_pose_.pose.orientation.w,
            current_pose_.pose.orientation.x,
            current_pose_.pose.orientation.y,
            current_pose_.pose.orientation.z
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
        
        // Update last velocities
        last_linear_vel_ = filtered_linear_vel_;
        last_angular_vel_ = filtered_angular_vel_;
        
        // Publish twist command
        geometry_msgs::msg::Twist twist_msg;
        twist_msg.linear.x = filtered_linear_vel_.x();
        twist_msg.linear.y = filtered_linear_vel_.y();
        twist_msg.linear.z = filtered_linear_vel_.z();
        twist_msg.angular.x = filtered_angular_vel_.x();
        twist_msg.angular.y = filtered_angular_vel_.y();
        twist_msg.angular.z = filtered_angular_vel_.z();
        
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