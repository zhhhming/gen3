#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_kdl/tf2_kdl.hpp>

#include <trac_ik/trac_ik.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include "xr_interfaces/srv/get_pose.hpp"
#include "xr_interfaces/srv/get_key_value.hpp"
#include "xr_interfaces/srv/get_button_state.hpp"

#include <Eigen/Dense>
#include <memory>
#include <chrono>
#include <deque>

using namespace std::chrono_literals;

class IKSolverNode : public rclcpp::Node
{
public:
    IKSolverNode() : Node("ik_solver_node")
    {
        // Declare parameters
        this->declare_parameter("robot_urdf_path", "/path/to/robot.urdf");
        this->declare_parameter("scale_factor", 1.0);
        this->declare_parameter("control_frequency", 50.0);
        this->declare_parameter("base_link", "base_link");
        this->declare_parameter("end_effector_link", "bracelet_link");
        this->declare_parameter("control_trigger", "right_grip");
        this->declare_parameter("gripper_trigger", "right_trigger");
        this->declare_parameter("pose_source", "right_controller");
        
        // IK parameters
        this->declare_parameter("ik_position_tolerance", 0.001);  // 1mm
        this->declare_parameter("ik_orientation_tolerance", 0.01); // ~0.57 degrees
        this->declare_parameter("ik_max_time", 0.005);  // 5ms max solve time
        this->declare_parameter("joint_smoothing_factor", 0.3);  // Exponential smoothing
        this->declare_parameter("max_joint_velocity", 2.0);  // rad/s
        
        // Get parameters
        urdf_path_ = this->get_parameter("robot_urdf_path").as_string();
        scale_factor_ = this->get_parameter("scale_factor").as_double();
        control_frequency_ = this->get_parameter("control_frequency").as_double();
        base_link_ = this->get_parameter("base_link").as_string();
        end_effector_link_ = this->get_parameter("end_effector_link").as_string();
        control_trigger_ = this->get_parameter("control_trigger").as_string();
        gripper_trigger_ = this->get_parameter("gripper_trigger").as_string();
        pose_source_ = this->get_parameter("pose_source").as_string();
        
        ik_pos_tolerance_ = this->get_parameter("ik_position_tolerance").as_double();
        ik_ori_tolerance_ = this->get_parameter("ik_orientation_tolerance").as_double();
        ik_max_time_ = this->get_parameter("ik_max_time").as_double();
        joint_smoothing_factor_ = this->get_parameter("joint_smoothing_factor").as_double();
        max_joint_velocity_ = this->get_parameter("max_joint_velocity").as_double();
        
        dt_ = 1.0 / control_frequency_;
        
        // Initialize TRAC-IK
        if (!initializeTracIK()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize TRAC-IK");
            throw std::runtime_error("TRAC-IK initialization failed");
        }
        
        // Initialize transforms
        initializeTransforms();
        
        // Create service clients
        xr_pose_client_ = this->create_client<xr_interfaces::srv::GetPose>("/xr/get_pose");
        xr_key_client_ = this->create_client<xr_interfaces::srv::GetKeyValue>("/xr/get_key_value");
        xr_button_client_ = this->create_client<xr_interfaces::srv::GetButtonState>("/xr/get_button_state");
        
        // Wait for services
        waitForServices();
        
        // Create subscribers
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&IKSolverNode::jointStateCallback, this, std::placeholders::_1));
        
        // Create publishers
        target_joint_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/target_joint_positions", 10);
        gripper_cmd_pub_ = this->create_publisher<std_msgs::msg::Float64>(
            "/gripper_command", 10);
        
        // Create timer
        auto period = std::chrono::duration<double>(dt_);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&IKSolverNode::controlLoop, this));
        
        RCLCPP_INFO(this->get_logger(), "IK Solver Node with TRAC-IK initialized");
        RCLCPP_INFO(this->get_logger(), "Position tolerance: %.3f mm", ik_pos_tolerance_ * 1000);
        RCLCPP_INFO(this->get_logger(), "Orientation tolerance: %.3f degrees", ik_ori_tolerance_ * 57.3);
    }
    
private:
    // TRAC-IK solver
    std::unique_ptr<TRAC_IK::TRAC_IK> tracik_solver_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    KDL::Chain kdl_chain_;
    
    // Joint state
    KDL::JntArray current_joint_positions_;
    KDL::JntArray target_joint_positions_;
    KDL::JntArray smoothed_joint_positions_;
    std::vector<std::string> joint_names_;
    std::map<std::string, int> joint_name_to_index_;
    int num_joints_;
    bool joints_initialized_ = false;
    
    // Solution history for consistency
    std::deque<KDL::JntArray> solution_history_;
    const size_t history_size_ = 5;
    
    // Parameters
    std::string urdf_path_, base_link_, end_effector_link_;
    std::string control_trigger_, gripper_trigger_, pose_source_;
    double scale_factor_;
    double control_frequency_, dt_;
    double ik_pos_tolerance_, ik_ori_tolerance_, ik_max_time_;
    double joint_smoothing_factor_, max_joint_velocity_;
    
    // Control state
    bool is_active_ = false;
    bool was_active_ = false;
    KDL::Frame ref_ee_frame_;
    KDL::Frame current_ee_frame_;
    Eigen::Vector3d ref_controller_pos_;
    Eigen::Quaterniond ref_controller_quat_;
    
    // Transform matrices
    Eigen::Matrix3d R_headset_world_;
    Eigen::Matrix3d R_z_90_cw_;
    
    // ROS interfaces
    rclcpp::Client<xr_interfaces::srv::GetPose>::SharedPtr xr_pose_client_;
    rclcpp::Client<xr_interfaces::srv::GetKeyValue>::SharedPtr xr_key_client_;
    rclcpp::Client<xr_interfaces::srv::GetButtonState>::SharedPtr xr_button_client_;
    
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr target_joint_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr gripper_cmd_pub_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    
    bool initializeTracIK()
    {
        // Read URDF file
        std::ifstream urdf_file(urdf_path_);
        if (!urdf_file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open URDF: %s", urdf_path_.c_str());
            return false;
        }
        
        std::string urdf_string((std::istreambuf_iterator<char>(urdf_file)),
                                std::istreambuf_iterator<char>());
        
        // Initialize TRAC-IK with Distance solve type for consistency
        tracik_solver_ = std::make_unique<TRAC_IK::TRAC_IK>(
            urdf_string,
            base_link_,
            end_effector_link_,
            ik_max_time_,
            ik_pos_tolerance_,
            TRAC_IK::Distance  // This prioritizes solutions close to current joint positions
        );
        
        // Get chain for FK solver
        KDL::Tree kdl_tree;
        if (!kdl_parser::treeFromString(urdf_string, kdl_tree)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF");
            return false;
        }
        
        if (!kdl_tree.getChain(base_link_, end_effector_link_, kdl_chain_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to extract chain");
            return false;
        }
        
        // Get joint limits and names
        KDL::JntArray ll, ul;
        bool valid = tracik_solver_->getKDLLimits(ll, ul);
        if (!valid) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get joint limits");
            return false;
        }
        
        num_joints_ = tracik_solver_->getNrOfJointsInChain();
        current_joint_positions_.resize(num_joints_);
        target_joint_positions_.resize(num_joints_);
        smoothed_joint_positions_.resize(num_joints_);
        
        // Initialize FK solver
        fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(kdl_chain_);
        
        // Get joint names
        for (unsigned int i = 0; i < kdl_chain_.getNrOfSegments(); ++i) {
            const KDL::Segment& segment = kdl_chain_.getSegment(i);
            const KDL::Joint& joint = segment.getJoint();
            if (joint.getType() != KDL::Joint::None) {
                joint_names_.push_back(joint.getName());
                joint_name_to_index_[joint.getName()] = joint_names_.size() - 1;
                RCLCPP_INFO(this->get_logger(), "Joint %zu: %s [%.2f, %.2f]", 
                           joint_names_.size()-1, joint.getName().c_str(),
                           ll(joint_names_.size()-1), ul(joint_names_.size()-1));
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "TRAC-IK initialized with %d joints", num_joints_);
        return true;
    }
    
    void initializeTransforms()
    {
        // R_HEADSET_TO_WORLD
        R_headset_world_ << 1, 0, 0,
                           0, 0, 1,
                           0, -1, 0;
        
        // 90 degree CW rotation around Z
        R_z_90_cw_ << 0, 1, 0,
                     -1, 0, 0,
                      0, 0, 1;
    }
    
    void waitForServices()
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for XR services...");
        
        while (!xr_pose_client_->wait_for_service(1s)) {
            if (!rclcpp::ok()) return;
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                "Waiting for XR services...");
        }
        
        RCLCPP_INFO(this->get_logger(), "XR services available");
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
        
        // Initialize smoothed positions on first callback
        if (!joints_initialized_) {
            smoothed_joint_positions_ = current_joint_positions_;
            target_joint_positions_ = current_joint_positions_;
            joints_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Joints initialized from joint_states");
        }
    }
    
    void processControllerPose(const std::vector<double>& xr_pose,
                              Eigen::Vector3d& delta_pos,
                              Eigen::Vector3d& delta_rot)
    {
        // Extract position and quaternion
        Eigen::Vector3d controller_pos(xr_pose[0], xr_pose[1], xr_pose[2]);
        Eigen::Quaterniond controller_quat(xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]);
        
        // Transform
        controller_pos = R_headset_world_ * controller_pos;
        Eigen::Quaterniond R_quat(R_headset_world_);
        controller_quat = R_quat * controller_quat * R_quat.conjugate();
        
        // Calculate deltas
        if (!was_active_) {
            ref_controller_pos_ = controller_pos;
            ref_controller_quat_ = controller_quat;
            delta_pos.setZero();
            delta_rot.setZero();
        } else {
            delta_pos = (controller_pos - ref_controller_pos_) * scale_factor_;
            
            // Angle-axis from quaternion difference
            Eigen::Quaterniond quat_diff = controller_quat * ref_controller_quat_.conjugate();
            Eigen::AngleAxisd angle_axis(quat_diff);
            delta_rot = angle_axis.angle() * angle_axis.axis();
        }
        
        // Apply 90° rotation
        delta_pos = R_z_90_cw_ * delta_pos;
        delta_rot = R_z_90_cw_ * delta_rot;
    }
    
    bool solveIKWithConsistency(const KDL::Frame& target_frame, KDL::JntArray& solution)
    {
        // Use current positions as seed for Distance solve type
        KDL::JntArray seed = smoothed_joint_positions_;
        
        // Solve IK with relaxed orientation tolerance
        int ret = tracik_solver_->CartToJnt(seed, target_frame, solution, 
                                           KDL::Twist(KDL::Vector(ik_pos_tolerance_, ik_pos_tolerance_, ik_pos_tolerance_),
                                                     KDL::Vector(ik_ori_tolerance_, ik_ori_tolerance_, ik_ori_tolerance_)));
        
        if (ret < 0) {
            // Try with more relaxed orientation if first attempt fails
            ret = tracik_solver_->CartToJnt(seed, target_frame, solution,
                                           KDL::Twist(KDL::Vector(ik_pos_tolerance_, ik_pos_tolerance_, ik_pos_tolerance_),
                                                     KDL::Vector(ik_ori_tolerance_*2, ik_ori_tolerance_*2, ik_ori_tolerance_*2)));
        }
        
        if (ret >= 0) {
            // Check for joint jumps
            bool has_jump = false;
            for (int i = 0; i < num_joints_; ++i) {
                double diff = std::abs(solution(i) - smoothed_joint_positions_(i));
                if (diff > M_PI) {  // Check for wrap-around
                    diff = 2*M_PI - diff;
                }
                if (diff > max_joint_velocity_ * dt_) {
                    has_jump = true;
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                       "Joint %d jump detected: %.3f rad", i, diff);
                }
            }
            
            // If jump detected, use smoothing to limit velocity
            if (has_jump) {
                for (int i = 0; i < num_joints_; ++i) {
                    double diff = solution(i) - smoothed_joint_positions_(i);
                    // Handle angle wrap
                    if (diff > M_PI) diff -= 2*M_PI;
                    if (diff < -M_PI) diff += 2*M_PI;
                    
                    // Limit velocity
                    double max_change = max_joint_velocity_ * dt_;
                    diff = std::max(-max_change, std::min(max_change, diff));
                    solution(i) = smoothed_joint_positions_(i) + diff;
                }
            }
            
            return true;
        }
        
        return false;
    }
    
    void smoothJointPositions(const KDL::JntArray& target)
    {
        // Exponential smoothing with angle wrap handling
        for (int i = 0; i < num_joints_; ++i) {
            double diff = target(i) - smoothed_joint_positions_(i);
            
            // Handle angle wrapping
            if (diff > M_PI) diff -= 2*M_PI;
            if (diff < -M_PI) diff += 2*M_PI;
            
            smoothed_joint_positions_(i) += joint_smoothing_factor_ * diff;
            
            // Normalize to [-pi, pi]
            while (smoothed_joint_positions_(i) > M_PI)
                smoothed_joint_positions_(i) -= 2*M_PI;
            while (smoothed_joint_positions_(i) < -M_PI)
                smoothed_joint_positions_(i) += 2*M_PI;
        }
    }
    
    void controlLoop()
    {
        if (!joints_initialized_) {
            RCLCPP_DEBUG(this->get_logger(), "Waiting for joint states...");
            return;
        }
        
        // Get grip value
        auto grip_request = std::make_shared<xr_interfaces::srv::GetKeyValue::Request>();
        grip_request->name = control_trigger_;
        auto grip_future = xr_key_client_->async_send_request(grip_request);
        
        if (grip_future.wait_for(10ms) != std::future_status::ready) {
            return;
        }
        
        auto grip_response = grip_future.get();
        if (!grip_response->success) return;
        
        is_active_ = (grip_response->value > 0.9);
        
        // Handle activation/deactivation
        if (is_active_ && !was_active_) {
            RCLCPP_INFO(this->get_logger(), "Control activated");
            
            // Get current end-effector pose
            fk_solver_->JntToCart(current_joint_positions_, current_ee_frame_);
            ref_ee_frame_ = current_ee_frame_;
            was_active_ = true;
            
            // Clear solution history
            solution_history_.clear();
            
        } else if (!is_active_ && was_active_) {
            RCLCPP_INFO(this->get_logger(), "Control deactivated");
            was_active_ = false;
        }
        
        // Process control
        if (is_active_) {
            // Get controller pose
            auto pose_request = std::make_shared<xr_interfaces::srv::GetPose::Request>();
            pose_request->name = pose_source_;
            auto pose_future = xr_pose_client_->async_send_request(pose_request);
            
            if (pose_future.wait_for(10ms) == std::future_status::ready) {
                auto pose_response = pose_future.get();
                
                if (pose_response->success && pose_response->pose.size() >= 7) {
                    // Calculate deltas
                    Eigen::Vector3d delta_pos, delta_rot;
                    processControllerPose(pose_response->pose, delta_pos, delta_rot);
                    
                    // Create target frame
                    KDL::Frame target_frame = ref_ee_frame_;
                    target_frame.p = target_frame.p + KDL::Vector(delta_pos[0], delta_pos[1], delta_pos[2]);
                    
                    // Apply rotation
                    double angle = delta_rot.norm();
                    if (angle > 1e-6) {
                        KDL::Vector axis(delta_rot[0]/angle, delta_rot[1]/angle, delta_rot[2]/angle);
                        KDL::Rotation delta_rotation = KDL::Rotation::Rot(axis, angle);
                        target_frame.M = delta_rotation * target_frame.M;
                    }
                    
                    // Solve IK with consistency check
                    KDL::JntArray ik_solution(num_joints_);
                    if (solveIKWithConsistency(target_frame, ik_solution)) {
                        // Smooth the solution
                        smoothJointPositions(ik_solution);
                        
                        // Publish target
                        sensor_msgs::msg::JointState target_msg;
                        target_msg.header.stamp = this->now();
                        target_msg.name = joint_names_;
                        target_msg.position.resize(num_joints_);
                        
                        for (int i = 0; i < num_joints_; ++i) {
                            target_msg.position[i] = smoothed_joint_positions_(i);
                        }
                        
                        target_joint_pub_->publish(target_msg);
                        
                        // Store in history
                        solution_history_.push_back(smoothed_joint_positions_);
                        if (solution_history_.size() > history_size_) {
                            solution_history_.pop_front();
                        }
                        
                    } else {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                           "IK solution not found");
                    }
                }
            }
        } else {
            // When not active, slowly move smoothed positions toward current
            smoothJointPositions(current_joint_positions_);
        }
        
        // Update gripper
        auto trigger_request = std::make_shared<xr_interfaces::srv::GetKeyValue::Request>();
        trigger_request->name = gripper_trigger_;
        auto trigger_future = xr_key_client_->async_send_request(trigger_request);
        
        if (trigger_future.wait_for(10ms) == std::future_status::ready) {
            auto trigger_response = trigger_future.get();
            if (trigger_response->success) {
                std_msgs::msg::Float64 gripper_msg;
                gripper_msg.data = trigger_response->value;
                gripper_cmd_pub_->publish(gripper_msg);
            }
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<IKSolverNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}