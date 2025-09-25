#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <urdf/model.h>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class ArmVisualizerNode : public rclcpp::Node
{
public:
    ArmVisualizerNode() : Node("arm_visualizer_node")
    {
        // 声明参数
        this->declare_parameter("robot_urdf_path", "/home/ming/xrrobotics_new/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/GEN3-7DOF.urdf");
        this->declare_parameter("base_link", "base_link");
        this->declare_parameter("current_prefix", "current_");
        this->declare_parameter("target_prefix", "target_");
        this->declare_parameter("publish_rate", 50.0);
        
        // 获取参数
        urdf_path_ = this->get_parameter("robot_urdf_path").as_string();
        base_link_ = this->get_parameter("base_link").as_string();
        current_prefix_ = this->get_parameter("current_prefix").as_string();
        target_prefix_ = this->get_parameter("target_prefix").as_string();
        publish_rate_ = this->get_parameter("publish_rate").as_double();
        
        // 初始化URDF和KDL
        if (!initializeURDFAndKDL()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize URDF and KDL");
            throw std::runtime_error("URDF/KDL initialization failed");
        }
        
        // 初始化TF广播器
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        
        // 创建订阅者
        current_joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&ArmVisualizerNode::currentJointStateCallback, this, std::placeholders::_1));
            
        target_joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/target_joint_positions", 10,
            std::bind(&ArmVisualizerNode::targetJointStateCallback, this, std::placeholders::_1));
        
        // 创建定时器用于发布TF
        auto period = std::chrono::duration<double>(1.0 / publish_rate_);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(period),
            std::bind(&ArmVisualizerNode::publishTransforms, this));
        
        RCLCPP_INFO(this->get_logger(), "Arm Visualizer Node initialized");
        RCLCPP_INFO(this->get_logger(), "Current arm prefix: '%s'", current_prefix_.c_str());
        RCLCPP_INFO(this->get_logger(), "Target arm prefix: '%s'", target_prefix_.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing rate: %.1f Hz", publish_rate_);
    }

private:
    // 参数
    std::string urdf_path_;
    std::string base_link_;
    std::string current_prefix_;
    std::string target_prefix_;
    double publish_rate_;
    
    // URDF和KDL相关
    urdf::Model urdf_model_;
    KDL::Tree kdl_tree_;
    std::map<std::string, KDL::Chain> joint_chains_;
    std::map<std::string, std::shared_ptr<KDL::ChainFkSolverPos_recursive>> fk_solvers_;
    std::vector<std::string> joint_names_;
    std::map<std::string, int> joint_name_to_index_;
    std::map<std::string, std::pair<double, double>> joint_limits_; // 存储关节限制
    
    // 检查并裁剪关节角度到限制范围
    double clampJointValue(const std::string& joint_name, double value)
    {
        auto it = joint_limits_.find(joint_name);
        if (it != joint_limits_.end()) {
            double lower = it->second.first;
            double upper = it->second.second;
            
            if (value < lower) {
                RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                     "Joint %s value %.3f below lower limit %.3f, clamping", 
                                     joint_name.c_str(), value, lower);
                return lower;
            } else if (value > upper) {
                RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                     "Joint %s value %.3f above upper limit %.3f, clamping", 
                                     joint_name.c_str(), value, upper);
                return upper;
            }
        }
        return value;
    }
    
    // 关节状态
    std::map<std::string, double> current_joint_positions_;
    std::map<std::string, double> target_joint_positions_;
    bool current_joints_received_ = false;
    bool target_joints_received_ = false;
    
    // ROS接口
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr current_joint_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr target_joint_sub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    bool initializeURDFAndKDL()
    {
        // 读取URDF文件
        std::ifstream urdf_file(urdf_path_);
        if (!urdf_file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open URDF: %s", urdf_path_.c_str());
            return false;
        }
        
        std::string urdf_string((std::istreambuf_iterator<char>(urdf_file)),
                               std::istreambuf_iterator<char>());
        urdf_file.close();
        
        // 解析URDF
        if (!urdf_model_.initString(urdf_string)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF");
            return false;
        }
        
        // 创建KDL树
        if (!kdl_parser::treeFromString(urdf_string, kdl_tree_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create KDL tree");
            return false;
        }
        
        // 获取所有关节信息和限制
        for (const auto& joint_pair : urdf_model_.joints_) {
            const auto& joint = joint_pair.second;
            if (joint->type != urdf::Joint::FIXED && joint->type != urdf::Joint::FLOATING) {
                joint_names_.push_back(joint->name);
                joint_name_to_index_[joint->name] = joint_names_.size() - 1;
                
                // 收集关节限制
                if (joint->limits) {
                    joint_limits_[joint->name] = std::make_pair(joint->limits->lower, joint->limits->upper);
                    RCLCPP_INFO(this->get_logger(), "Joint %s: limits [%.3f, %.3f] (type: %d)", 
                               joint->name.c_str(), joint->limits->lower, joint->limits->upper, joint->type);
                } else {
                    // 对于没有限制的关节，使用默认值
                    joint_limits_[joint->name] = std::make_pair(-M_PI, M_PI);
                    RCLCPP_INFO(this->get_logger(), "Joint %s: no limits, using [-π, π] (type: %d)", 
                               joint->name.c_str(), joint->type);
                }
            }
        }
        
        // 只为每个连杆创建一次FK链，并验证链的有效性
        for (const auto& link_pair : urdf_model_.links_) {
            const std::string& link_name = link_pair.first;
            if (link_name != base_link_) {
                KDL::Chain chain;
                if (kdl_tree_.getChain(base_link_, link_name, chain)) {
                    // 验证链的结构
                    bool chain_valid = true;
                    RCLCPP_INFO(this->get_logger(), "Chain %s -> %s has %d segments:", 
                               base_link_.c_str(), link_name.c_str(), chain.getNrOfSegments());
                    
                    int active_joints = 0;
                    for (unsigned int i = 0; i < chain.getNrOfSegments(); ++i) {
                        const KDL::Segment& segment = chain.getSegment(i);
                        const KDL::Joint& joint = segment.getJoint();
                        
                        RCLCPP_INFO(this->get_logger(), "  Segment %d: %s, Joint: %s (type: %d)", 
                                   i, segment.getName().c_str(), joint.getName().c_str(), joint.getType());
                        
                        if (joint.getType() != KDL::Joint::None) {
                            active_joints++;
                        }
                    }
                    
                    RCLCPP_INFO(this->get_logger(), "  Active joints: %d", active_joints);
                    
                    if (active_joints > 0) {
                        joint_chains_[link_name] = chain;
                        fk_solvers_[link_name] = std::make_shared<KDL::ChainFkSolverPos_recursive>(chain);
                        
                        // 测试FK求解器，使用零位置
                        KDL::JntArray zero_joints(active_joints);
                        KDL::Frame test_frame;
                        for (int j = 0; j < active_joints; j++) {
                            zero_joints(j) = 0.0;
                        }
                        
                        int test_result = fk_solvers_[link_name]->JntToCart(zero_joints, test_frame);
                        if (test_result >= 0) {
                            RCLCPP_INFO(this->get_logger(), "  FK solver test PASSED for %s", link_name.c_str());
                        } else {
                            RCLCPP_WARN(this->get_logger(), "  FK solver test FAILED for %s (error: %d)", 
                                       link_name.c_str(), test_result);
                        }
                    } else {
                        RCLCPP_DEBUG(this->get_logger(), "Skipping %s - no active joints", link_name.c_str());
                    }
                } else {
                    RCLCPP_WARN(this->get_logger(), "Failed to get chain: %s -> %s", 
                               base_link_.c_str(), link_name.c_str());
                }
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Initialized: %zu joints, %zu links with FK chains", 
                   joint_names_.size(), joint_chains_.size());
        
        return true;
    }
    
    void currentJointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        updateJointPositions(msg, current_joint_positions_);
        current_joints_received_ = true;
    }
    
    void targetJointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        updateJointPositions(msg, target_joint_positions_);
        target_joints_received_ = true;
    }
    
    void updateJointPositions(const sensor_msgs::msg::JointState::SharedPtr msg, 
                             std::map<std::string, double>& joint_map)
    {
        for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
            joint_map[msg->name[i]] = msg->position[i];
        }
    }
    
    void publishTransforms()
    {
        if (!current_joints_received_ && !target_joints_received_) {
            RCLCPP_DEBUG(this->get_logger(), "No joint state data received yet");
            return;
        }
        
        rclcpp::Time now = this->now();
        std::vector<geometry_msgs::msg::TransformStamped> transforms;
        
        // 发布当前位置的TF
        if (current_joints_received_) {
            publishArmTransforms(current_joint_positions_, current_prefix_, now, transforms);
        }
        
        // 发布目标位置的TF
        if (target_joints_received_) {
            publishArmTransforms(target_joint_positions_, target_prefix_, now, transforms);
        }
        
        // 批量发布所有变换
        if (!transforms.empty()) {
            tf_broadcaster_->sendTransform(transforms);
        }
    }
    
    void publishArmTransforms(const std::map<std::string, double>& joint_positions,
                             const std::string& prefix,
                             const rclcpp::Time& timestamp,
                             std::vector<geometry_msgs::msg::TransformStamped>& transforms)
    {
        // 首先发布base_link
        geometry_msgs::msg::TransformStamped base_transform;
        base_transform.header.stamp = timestamp;
        base_transform.header.frame_id = "world";
        base_transform.child_frame_id = prefix + base_link_;
        base_transform.transform.translation.x = 0.0;
        base_transform.transform.translation.y = 0.0;
        base_transform.transform.translation.z = 0.0;
        base_transform.transform.rotation.x = 0.0;
        base_transform.transform.rotation.y = 0.0;
        base_transform.transform.rotation.z = 0.0;
        base_transform.transform.rotation.w = 1.0;
        transforms.push_back(base_transform);
        
        // 为每个link计算并发布变换
        for (const auto& chain_pair : joint_chains_) {
            const std::string& link_name = chain_pair.first;
            const KDL::Chain& chain = chain_pair.second;
            
            // 准备关节数组
            KDL::JntArray joint_array(chain.getNrOfJoints());
            int joint_idx = 0;
            
            // 填充关节位置并进行详细调试
            for (unsigned int i = 0; i < chain.getNrOfSegments(); ++i) {
                const KDL::Segment& segment = chain.getSegment(i);
                const KDL::Joint& joint = segment.getJoint();
                
                if (joint.getType() != KDL::Joint::None) {
                    std::string joint_name = joint.getName();
                    auto it = joint_positions.find(joint_name);
                    if (it != joint_positions.end()) {
                        // 检查数值有效性
                        if (std::isnan(it->second) || std::isinf(it->second)) {
                            RCLCPP_WARN(this->get_logger(), "Invalid joint value for %s: %f", 
                                       joint_name.c_str(), it->second);
                            joint_array(joint_idx) = 0.0;
                        } else {
                            // 应用关节限制裁剪
                            double clamped_value = clampJointValue(joint_name, it->second);
                            joint_array(joint_idx) = clamped_value;
                            
                            // 记录原始值和裁剪值的差异
                            if (std::abs(clamped_value - it->second) > 1e-6) {
                                RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                                     "Joint %s: %.3f -> %.3f (clamped)", 
                                                     joint_name.c_str(), it->second, clamped_value);
                            }
                        }
                    } else {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                            "Joint %s not found in joint_positions for link %s", 
                                            joint_name.c_str(), link_name.c_str());
                        joint_array(joint_idx) = 0.0;
                    }
                    joint_idx++;
                }
            }
            
            // 检查FK求解器
            auto it_fk = fk_solvers_.find(link_name);
            if (it_fk == fk_solvers_.end() || !it_fk->second) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                    "No FK solver for link '%s'", link_name.c_str());
                continue;
            }
            
            // 计算正向运动学
            KDL::Frame result_frame;
            auto& fk_solver = it_fk->second;
            int fk_result = fk_solver->JntToCart(joint_array, result_frame);
            
            if (fk_result >= 0) {
                // 创建变换消息
                geometry_msgs::msg::TransformStamped transform;
                transform.header.stamp = timestamp;
                transform.header.frame_id = prefix + base_link_;
                transform.child_frame_id = prefix + link_name;
                
                // 设置平移
                transform.transform.translation.x = result_frame.p.x();
                transform.transform.translation.y = result_frame.p.y();
                transform.transform.translation.z = result_frame.p.z();
                
                // 设置旋转
                double x, y, z, w;
                result_frame.M.GetQuaternion(x, y, z, w);
                transform.transform.rotation.x = x;
                transform.transform.rotation.y = y;
                transform.transform.rotation.z = z;
                transform.transform.rotation.w = w;
                
                transforms.push_back(transform);
                
                RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "FK success for %s: pos[%.3f,%.3f,%.3f]", 
                                     link_name.c_str(), result_frame.p.x(), result_frame.p.y(), result_frame.p.z());
            } else {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                    "FK failed for link '%s' with error code: %d", 
                                    link_name.c_str(), fk_result);
                
                // 打印详细的调试信息
                RCLCPP_DEBUG(this->get_logger(), "Chain for '%s' has %d joints:", 
                           link_name.c_str(), chain.getNrOfJoints());
                for (int j = 0; j < joint_array.rows(); j++) {
                    RCLCPP_DEBUG(this->get_logger(), "  Joint %d: %.6f", j, joint_array(j));
                }
            }
        }
        
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                             "[%s] Publishing %zu TFs (including base)", prefix.c_str(), transforms.size());
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<ArmVisualizerNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}