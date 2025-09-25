#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <fstream>
#include <map>
#include <memory>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "xr_interfaces/msg/key_value.hpp"
#include "xr_interfaces/msg/pose.hpp"

using namespace std::chrono_literals;

class CartesianTargetNode : public rclcpp::Node
{
public:
    CartesianTargetNode() : Node("cartesian_target_node")
    {
        // Declare parameters
        this->declare_parameter("robot_urdf_path", "/home/ming/xrrobotics_new/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/GEN3-7DOF.urdf");
        this->declare_parameter("scale_factor", 1.0);
        this->declare_parameter("control_frequency", 100.0);
        this->declare_parameter("base_link", "base_link");
        this->declare_parameter("end_effector_link", "bracelet_link");
        
        // Get parameters
        urdf_path_ = this->get_parameter("robot_urdf_path").as_string();
        scale_factor_ = this->get_parameter("scale_factor").as_double();
        control_frequency_ = this->get_parameter("control_frequency").as_double();
        base_link_ = this->get_parameter("base_link").as_string();
        end_effector_link_ = this->get_parameter("end_effector_link").as_string();
        
        dt_ = 1.0 / control_frequency_;
        
        // Initialize FK solver
        if (!initializeFKSolver()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize FK solver");
            throw std::runtime_error("FK solver initialization failed");
        }
        
        // Initialize transforms
        initializeTransforms();
        
        // Create subscribers for XR data
        right_grip_sub_ = this->create_subscription<xr_interfaces::msg::KeyValue>(
            "/xr/right_grip", 10,
            std::bind(&CartesianTargetNode::rightGripCallback, this, std::placeholders::_1));
        
        right_trigger_sub_ = this->create_subscription<xr_interfaces::msg::KeyValue>(
            "/xr/right_trigger", 10,
            std::bind(&CartesianTargetNode::rightTriggerCallback, this, std::placeholders::_1));
        
        right_controller_pose_sub_ = this->create_subscription<xr_interfaces::msg::Pose>(
            "/xr/right_controller_pose", 10,
            std::bind(&CartesianTargetNode::rightControllerPoseCallback, this, std::placeholders::_1));
        
        // Create subscriber for joint states
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&CartesianTargetNode::jointStateCallback, this, std::placeholders::_1));
        
        // Create publishers
        target_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/cartesian_target", 10);
        
        current_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/current_ee_pose", 10);
            
        gripper_cmd_pub_ = this->create_publisher<std_msgs::msg::Float64>(
            "/gripper_command", 10);
        
        // Wait for topics
        waitForTopics();
        
        // Create timer
        auto period = std::chrono::duration<double>(dt_);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&CartesianTargetNode::controlLoop, this));
        
        RCLCPP_INFO(this->get_logger(), "Cartesian Target Node initialized");
        RCLCPP_INFO(this->get_logger(), "Scale factor: %.2f", scale_factor_);
        RCLCPP_INFO(this->get_logger(), "Control frequency: %.1f Hz", control_frequency_);
    }
    
private:
    // FK solver
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    KDL::Chain kdl_chain_;
    
    // Joint state
    KDL::JntArray current_joint_positions_;
    std::vector<std::string> joint_names_;
    std::map<std::string, int> joint_name_to_index_;
    int num_joints_;
    bool joints_initialized_ = false;
    
    // Current end-effector pose
    KDL::Frame current_ee_frame_;
    
    // XR data storage
    double current_grip_value_ = 0.0;
    double current_trigger_value_ = 0.0;
    std::vector<double> current_controller_pose_;
    bool xr_data_received_ = false;
    
    // Parameters
    std::string urdf_path_, base_link_, end_effector_link_;
    double scale_factor_;
    double control_frequency_, dt_;
    
    // Control state
    bool is_active_ = false;
    KDL::Frame ref_ee_frame_;
    bool ref_ee_frame_valid_ = false;
    Eigen::Vector3d ref_controller_pos_;
    Eigen::Quaterniond ref_controller_quat_;
    bool ref_controller_valid_ = false;
    
    // Transform matrices
    Eigen::Matrix3d R_headset_world_;
    Eigen::Matrix3d R_z_90_cw_;
    
    // ROS interfaces
    rclcpp::Subscription<xr_interfaces::msg::KeyValue>::SharedPtr right_grip_sub_;
    rclcpp::Subscription<xr_interfaces::msg::KeyValue>::SharedPtr right_trigger_sub_;
    rclcpp::Subscription<xr_interfaces::msg::Pose>::SharedPtr right_controller_pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr current_pose_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr gripper_cmd_pub_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    
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
            RCLCPP_ERROR(this->get_logger(), "Failed to extract chain from %s to %s", 
                         base_link_.c_str(), end_effector_link_.c_str());
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
                RCLCPP_INFO(this->get_logger(), "Joint %zu: %s", 
                           joint_names_.size()-1, joint.getName().c_str());
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "FK solver initialized with %d joints", num_joints_);
        return true;
    }
    
    void initializeTransforms()
    {
        // R_HEADSET_TO_WORLD (same as IK node)
        R_headset_world_ << 0, 0, -1,
                           -1, 0, 0,
                           0, 1, 0;
        
        // 90 degree CW rotation around Z
        R_z_90_cw_ << 0, 1, 0,
                     -1, 0, 0,
                      0, 0, 1;
    }
    
    void waitForTopics()
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for XR topics...");
        
        while (rclcpp::ok()) {
            auto grip_count = this->count_publishers("/xr/right_grip");
            auto trigger_count = this->count_publishers("/xr/right_trigger");
            auto pose_count = this->count_publishers("/xr/right_controller_pose");
            
            if (grip_count > 0 && trigger_count > 0 && pose_count > 0) {
                RCLCPP_INFO(this->get_logger(), "All XR topics are available");
                break;
            }
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                "Waiting for XR topics: grip=%ld, trigger=%ld, pose=%ld",
                                grip_count, trigger_count, pose_count);
            
            rclcpp::sleep_for(100ms);
        }
    }
    
    // XR data callbacks
    void rightGripCallback(const xr_interfaces::msg::KeyValue::SharedPtr msg)
    {
        current_grip_value_ = msg->value;
        xr_data_received_ = true;
    }
    
    void rightTriggerCallback(const xr_interfaces::msg::KeyValue::SharedPtr msg)
    {
        current_trigger_value_ = msg->value;
    }
    
    void rightControllerPoseCallback(const xr_interfaces::msg::Pose::SharedPtr msg)
    {
        if (msg->pose.size() >= 7) {
            current_controller_pose_ = msg->pose;
        }
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
        
        // Initialize on first callback
        if (!joints_initialized_) {
            joints_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Joints initialized from joint_states");
        }
        
        // Compute current end-effector pose
        if (joints_initialized_) {
            fk_solver_->JntToCart(current_joint_positions_, current_ee_frame_);
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
        if (!ref_controller_valid_) {
            ref_controller_pos_ = controller_pos;
            ref_controller_quat_ = controller_quat;
            ref_controller_valid_ = true;
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
    
    void controlLoop()
    {
        if (!joints_initialized_ || !xr_data_received_) {
            RCLCPP_DEBUG(this->get_logger(), "Waiting for joint states and XR data...");
            return;
        }
        
        // Publish current end-effector pose
        auto current_pose_msg = kdlFrameToPoseMsg(current_ee_frame_);
        current_pose_pub_->publish(current_pose_msg);
        
        // Check activation state
        bool new_active = (current_grip_value_ > 0.9);
        
        if (new_active != is_active_) {
            if (new_active) {
                RCLCPP_INFO(this->get_logger(), "Control activated");
                
                // Initialize reference frames
                ref_ee_frame_ = current_ee_frame_;
                ref_ee_frame_valid_ = true;
                ref_controller_valid_ = false;  // Force reinitialize controller reference
                
            } else {
                RCLCPP_INFO(this->get_logger(), "Control deactivated");
                
                // Clear references
                ref_ee_frame_valid_ = false;
                ref_controller_valid_ = false;
            }
            
            is_active_ = new_active;
        }
        
        // Calculate and publish target pose if active
        if (is_active_ && current_controller_pose_.size() >= 7) {
            // Calculate deltas
            Eigen::Vector3d delta_pos, delta_rot;
            processControllerPose(current_controller_pose_, delta_pos, delta_rot);
            
            if (ref_ee_frame_valid_) {
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
                
                // Publish target pose
                auto target_pose_msg = kdlFrameToPoseMsg(target_frame);
                target_pose_pub_->publish(target_pose_msg);
            }
        } else if (!is_active_) {
            // When not active, publish current pose as target
            auto target_pose_msg = kdlFrameToPoseMsg(current_ee_frame_);
            target_pose_pub_->publish(target_pose_msg);
        }
        
        // Publish gripper command
        std_msgs::msg::Float64 gripper_msg;
        gripper_msg.data = current_trigger_value_;
        gripper_cmd_pub_->publish(gripper_msg);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<CartesianTargetNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}