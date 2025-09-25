#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <cmath>  // 用于M_PI

using namespace std::chrono_literals;

class TrajectoryFilterNode : public rclcpp::Node
{
public:
    TrajectoryFilterNode() : Node("trajectory_filter_node")
    {
        // Declare parameters
        this->declare_parameter("filter_alpha", 0.15);  // 低通滤波系数 (0-1)
        this->declare_parameter("control_rate", 100.0);  // 控制频率Hz
        this->declare_parameter("filter_deadband", 0.001);  // 死区阈值(弧度)
        this->declare_parameter("trajectory_duration", 0.1);  // 轨迹持续时间(秒)
        
        // Get parameters
        filter_alpha_ = this->get_parameter("filter_alpha").as_double();
        control_rate_ = this->get_parameter("control_rate").as_double();
        filter_deadband_ = this->get_parameter("filter_deadband").as_double();
        trajectory_duration_ = this->get_parameter("trajectory_duration").as_double();
        
        dt_ = 1.0 / control_rate_;
        
        // 定义关节名称 (根据你的机器人配置调整)
        joint_names_ = {
            "joint_1", "joint_2", "joint_3", "joint_4", 
            "joint_5", "joint_6", "joint_7","right_finger_bottom_joint"
        };
        
        num_joints_ = joint_names_.size();
        
        // 初始化滤波状态
        filtered_positions_.resize(num_joints_, 0.0);
        target_positions_.resize(num_joints_, 0.0);
        current_positions_.resize(num_joints_, 0.0);
        
        // 创建索引映射
        for (size_t i = 0; i < joint_names_.size(); ++i) {
            joint_name_to_index_[joint_names_[i]] = i;
        }
        
        // Create subscribers
        target_joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/target_joint_positions", 10,
            std::bind(&TrajectoryFilterNode::targetJointCallback, this, std::placeholders::_1));
        
        gripper_cmd_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/gripper_command", 10,
            std::bind(&TrajectoryFilterNode::gripperCommandCallback, this, std::placeholders::_1));
        
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&TrajectoryFilterNode::jointStateCallback, this, std::placeholders::_1));
        
        // Create publisher
        trajectory_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/joint_trajectory_controller/joint_trajectory", 10);
        
        // Wait for initial data
        waitForInitialData();
        
        // Create control timer
        auto period = std::chrono::duration<double>(dt_);
        control_timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&TrajectoryFilterNode::controlLoop, this));
        
        RCLCPP_INFO(this->get_logger(), "Trajectory Filter Node initialized");
        RCLCPP_INFO(this->get_logger(), "Filter alpha: %.3f", filter_alpha_);
        RCLCPP_INFO(this->get_logger(), "Control rate: %.1f Hz", control_rate_);
        RCLCPP_INFO(this->get_logger(), "Deadband: %.4f rad", filter_deadband_);
        RCLCPP_INFO(this->get_logger(), "Trajectory duration: %.3f sec", trajectory_duration_);
    }
    
private:
    // Parameters
    double filter_alpha_;
    double control_rate_, dt_;
    double filter_deadband_;
    double trajectory_duration_;
    
    // Joint configuration
    std::vector<std::string> joint_names_;
    std::map<std::string, size_t> joint_name_to_index_;
    size_t num_joints_;
    
    // State variables
    std::vector<double> filtered_positions_;
    std::vector<double> target_positions_;
    std::vector<double> current_positions_;
    double gripper_command_ = 0.0;
    
    // Initialization flags
    bool target_received_ = false;
    bool joint_states_received_ = false;
    bool gripper_received_ = false;
    bool initialized_ = false;
    
    // ROS interfaces
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr target_joint_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr gripper_cmd_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    
    void waitForInitialData()
    {
        RCLCPP_INFO(this->get_logger(), "Waiting for initial data...");
        
        auto start_time = this->now();
        while (rclcpp::ok() && (!target_received_ || !joint_states_received_ || !gripper_received_)) {
            
            auto elapsed = (this->now() - start_time).seconds();
            if (elapsed > 10.0) {
                RCLCPP_WARN(this->get_logger(), 
                           "Timeout waiting for initial data. target=%d, joints=%d, gripper=%d", 
                           target_received_, joint_states_received_, gripper_received_);
                break;
            }
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "Waiting... target=%d, joints=%d, gripper=%d",
                               target_received_, joint_states_received_, gripper_received_);
            
            rclcpp::sleep_for(100ms);
        }
        
        if (target_received_ && joint_states_received_) {
            // 用当前位置初始化滤波器
            filtered_positions_ = current_positions_;
            initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Filter initialized with current joint positions");
        }
    }
    
    void targetJointCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        if (msg->name.size() != msg->position.size()) {
            RCLCPP_WARN(this->get_logger(), "Joint names and positions size mismatch");
            return;
        }
        
        // 更新目标位置
        for (size_t i = 0; i < msg->name.size(); ++i) {
            auto it = joint_name_to_index_.find(msg->name[i]);
            if (it != joint_name_to_index_.end()) {
                target_positions_[it->second] = msg->position[i];
            }
        }
        
        target_received_ = true;
    }
    
    void gripperCommandCallback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        gripper_command_ = msg->data;
        gripper_received_ = true;
    }
    
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        if (msg->name.size() != msg->position.size()) {
            RCLCPP_WARN(this->get_logger(), "Joint states size mismatch");
            return;
        }
        
        // 更新当前位置
        for (size_t i = 0; i < msg->name.size(); ++i) {
            auto it = joint_name_to_index_.find(msg->name[i]);
            if (it != joint_name_to_index_.end()) {
                current_positions_[it->second] = msg->position[i];
            }
        }
        
        joint_states_received_ = true;
    }
    
    double normalizeAngleRad(double angle_rad) const
    {
        // 将角度归一化到 [-π, π] 范围
        while (angle_rad > M_PI) {
            angle_rad -= 2.0 * M_PI;
        }
        while (angle_rad < -M_PI) {
            angle_rad += 2.0 * M_PI;
        }
        return angle_rad;
    }
    
    double toNearestEquivalentAngle(double target_rad, double current_rad) const
    {
        // 将目标角度调整为离当前角度最近的等效角度
        // 例如：current=3.0, target=-3.1 -> 返回 3.28 (而不是 -3.1)
        
        double diff = target_rad - current_rad;
        
        // 归一化差值到 [-π, π]
        diff = normalizeAngleRad(diff);
        
        // 返回最近的等效角度
        return current_rad + diff;
    }
    
    void applyLowPassFilter()
    {
        for (size_t i = 0; i < num_joints_-1; ++i) {
            // 角度连续性处理：将目标角度调整为最近的等效角度
            double adjusted_target = toNearestEquivalentAngle(target_positions_[i], filtered_positions_[i]);
            
            double error = adjusted_target - filtered_positions_[i];
            
            // 死区处理
            if (std::abs(error) < filter_deadband_) {
                continue; // 在死区内，不更新
            }
            
            // 低通滤波：y[n] = alpha * x[n] + (1-alpha) * y[n-1]
            filtered_positions_[i] = filter_alpha_ * adjusted_target + 
                                   (1.0 - filter_alpha_) * filtered_positions_[i];
            
            // 确保输出角度在 [-π, π] 范围内
            filtered_positions_[i] = normalizeAngleRad(filtered_positions_[i]);
        }
    }
    
    void publishTrajectory()
    {
        trajectory_msgs::msg::JointTrajectory trajectory_msg;
        trajectory_msg.header.stamp = this->now();
        trajectory_msg.joint_names = joint_names_;
        
        // 创建轨迹点
        trajectory_msgs::msg::JointTrajectoryPoint point;
        
        // 设置位置 (前6个关节 + 夹爪)
        point.positions.resize(num_joints_);
        for (size_t i = 0; i < num_joints_ - 1; ++i) {  // 前6个关节使用滤波后的位置
            point.positions[i] = filtered_positions_[i];
        }
        
        // 夹爪位置不滤波，直接使用命令值
        point.positions[num_joints_ - 1] = gripper_command_;
        
        // 设置时间
        point.time_from_start = rclcpp::Duration::from_seconds(trajectory_duration_);
        
        trajectory_msg.points.push_back(point);
        
        // 发布轨迹
        trajectory_pub_->publish(trajectory_msg);
        
        RCLCPP_DEBUG(this->get_logger(), "Published trajectory with %zu joints", num_joints_);
    }
    
    void controlLoop()
    {
        if (!initialized_) {
            RCLCPP_DEBUG(this->get_logger(), "Waiting for initialization...");
            return;
        }
        
        // 应用低通滤波
        applyLowPassFilter();
        
        // 发布轨迹
        publishTrajectory();
        
        // 可选：打印调试信息
        static int counter = 0;
        if (++counter % static_cast<int>(control_rate_) == 0) {  // 每秒打印一次
            RCLCPP_DEBUG(this->get_logger(), "Filter running at %.1f Hz, gripper: %.3f", 
                        control_rate_, gripper_command_);
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<TrajectoryFilterNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}