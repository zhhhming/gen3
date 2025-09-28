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
#include <filesystem>
#include <iomanip>
#include <cstdlib>
#include <algorithm>  
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
        this->declare_parameter("gripper_control_frequency", 10.0);
        this->declare_parameter("base_link", "base_link");
        this->declare_parameter("end_effector_link", "bracelet_link");
        
        // PD控制参数
        this->declare_parameter("position_kp", 1.0);      // 位置比例增益
        this->declare_parameter("position_kd", 8.0);      // 位置微分增益
        this->declare_parameter("orientation_kp",1.0);   // 姿态比例增益
        this->declare_parameter("orientation_kd", 0.7);   // 姿态微分增益
        
        // 速度和加速度限制
        this->declare_parameter("max_linear_velocity", 0.5);      
        this->declare_parameter("max_angular_velocity", 1.0);     
        this->declare_parameter("max_linear_acceleration", 2.0);  
        this->declare_parameter("max_angular_acceleration", 5.0); 
        
        // 改进的滤波参数
        this->declare_parameter("target_filter_alpha", 1.0);      // 目标极轻滤波（快速响应）
        this->declare_parameter("current_filter_alpha", 1.0);      // 当前位姿轻滤波（减少测量噪音）
        this->declare_parameter("error_filter_alpha", 0.3);        // 误差滤波（用于比例项和微分项）
        this->declare_parameter("velocity_filter_alpha", 0.2);     // 速度输出滤波（可以更轻）
        
        // 速度死区参数（只保留速度死区）
        this->declare_parameter("velocity_deadzone", 0.003);      // 速度死区
        
        // 微分项尖峰检测参数
        this->declare_parameter("error_near_zero_threshold", 0.005);  // 误差接近0的阈值
        this->declare_parameter("spike_detection_factor", 5.0);       // 尖峰检测因子
        
        this->declare_parameter("publish_angular_in_degrees", true);
        this->declare_parameter("log_enabled", true);
        this->declare_parameter("log_dir", std::string("./vc_logs"));
        this->declare_parameter("plot_script", std::string("/home/ming/ros2_ws/vc_logs/plot_vc_log.py"));
        this->declare_parameter("python_interpreter", std::string("/home/ming/miniconda3/envs/xrrobotics/bin/python"));
        
        python_interpreter_ = this->get_parameter("python_interpreter").as_string();
        plot_script_ = this->get_parameter("plot_script").as_string();

        // Get parameters
        urdf_path_ = this->get_parameter("robot_urdf_path").as_string();
        control_frequency_ = this->get_parameter("control_frequency").as_double();
        gripper_control_frequency_ = this->get_parameter("gripper_control_frequency").as_double();
        base_link_ = this->get_parameter("base_link").as_string();
        end_effector_link_ = this->get_parameter("end_effector_link").as_string();
        
        // PD控制增益
        position_kp_ = this->get_parameter("position_kp").as_double();
        position_kd_ = this->get_parameter("position_kd").as_double();
        orientation_kp_ = this->get_parameter("orientation_kp").as_double();
        orientation_kd_ = this->get_parameter("orientation_kd").as_double();
        
        // 速度和加速度限制
        max_linear_vel_ = this->get_parameter("max_linear_velocity").as_double();
        max_angular_vel_ = this->get_parameter("max_angular_velocity").as_double();
        max_linear_acc_ = this->get_parameter("max_linear_acceleration").as_double();
        max_angular_acc_ = this->get_parameter("max_angular_acceleration").as_double();
        
        // 改进的滤波参数
        target_filter_alpha_ = this->get_parameter("target_filter_alpha").as_double();
        current_filter_alpha_ = this->get_parameter("current_filter_alpha").as_double();
        error_filter_alpha_ = this->get_parameter("error_filter_alpha").as_double();
        filter_alpha_ = this->get_parameter("velocity_filter_alpha").as_double();
        
        // 速度死区
        velocity_deadzone_ = this->get_parameter("velocity_deadzone").as_double();
        
        // 微分项尖峰检测参数
        error_near_zero_threshold_ = this->get_parameter("error_near_zero_threshold").as_double();
        spike_detection_factor_ = this->get_parameter("spike_detection_factor").as_double();
        
        publish_ang_in_degrees_ = this->get_parameter("publish_angular_in_degrees").as_bool();
        log_enabled_ = this->get_parameter("log_enabled").as_bool();
        log_dir_     = this->get_parameter("log_dir").as_string();

        R_ctrl_from_rviz_ << 0, 1, 0,
                             0, 0, 1,
                             1, 0, 0;
        dt_ = 1.0 / control_frequency_;
        
        // Initialize FK solver
        if (!initializeFKSolver()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize FK solver");
            throw std::runtime_error("FK solver initialization failed");
        }
        
        // 初始化控制状态
        filtered_linear_vel_.setZero();
        filtered_angular_vel_.setZero();
        last_linear_vel_.setZero();
        last_angular_vel_.setZero();
        
        // 初始化滤波后的位姿和误差
        filtered_target_position_.setZero();
        filtered_target_quaternion_ = Eigen::Quaterniond::Identity();
        filtered_current_position_.setZero();
        filtered_current_quaternion_ = Eigen::Quaterniond::Identity();
        filtered_position_error_.setZero();
        filtered_orientation_error_.setZero();
        
        // 初始化PD控制相关变量
        last_filtered_position_error_.setZero();
        last_filtered_orientation_error_.setZero();
        first_control_cycle_ = true;
        target_initialized_ = false;
        current_initialized_ = false;
        
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
                this->current_gripper_goal_ = handle;
                RCLCPP_DEBUG(this->get_logger(), "Gripper goal accepted");
            }
        };

        send_goal_options_.feedback_callback =
        [this](rclcpp_action::ClientGoalHandle<GripperCommand>::SharedPtr /*handle*/,
                const std::shared_ptr<const GripperCommand::Feedback> feedback)
        {
            // 可选：打印/使用反馈
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
        
        start_time_ = this->now();
        if (log_enabled_) {
            try {
                std::filesystem::create_directories(log_dir_);
                RCLCPP_INFO(this->get_logger(), "Logging enabled. Directory: %s", log_dir_.c_str());
            } catch (const std::exception& e) {
                RCLCPP_WARN(this->get_logger(), "Failed to create log dir: %s (logging disabled)", e.what());
                log_enabled_ = false;
            }
        }
        RCLCPP_INFO(this->get_logger(), "PD Velocity Control Node initialized with improved filtering");
        RCLCPP_INFO(this->get_logger(), "Filter alphas - Target: %.2f, Current: %.2f, Error: %.2f, Velocity: %.2f",
                   target_filter_alpha_, current_filter_alpha_, error_filter_alpha_, filter_alpha_);
        RCLCPP_INFO(this->get_logger(), "Position: Kp=%.2f, Kd=%.2f | Orientation: Kp=%.2f, Kd=%.2f", 
                   position_kp_, position_kd_, orientation_kp_, orientation_kd_);
    }
    
    void saveLogsToCsvAndPlots()
    {
        if (!log_enabled_ || logs_.empty()) return;

        // === 1) 写 CSV ===
        std::string csv_path = (std::filesystem::path(log_dir_) / "vc_log.csv").string();
        try {
            std::ofstream ofs(csv_path);
            ofs << std::fixed << std::setprecision(6);
            // 表头
            ofs << "t,"
                // pose
                << "pos_x,pos_y,pos_z,"
                << "quat_w,quat_x,quat_y,quat_z,"
                << "aa_x,aa_y,aa_z,"

                // error
                << "err_pos_x,err_pos_y,err_pos_z,"
                << "err_ori_x,err_ori_y,err_ori_z,"

                // derivatives (raw)
                << "der_pos_raw_x,der_pos_raw_y,der_pos_raw_z,"
                << "der_ori_raw_x,der_ori_raw_y,der_ori_raw_z,"

                // PD terms
                << "lin_p_x,lin_p_y,lin_p_z,"
                << "lin_d_x,lin_d_y,lin_d_z,"
                << "ang_p_x,ang_p_y,ang_p_z,"
                << "ang_d_x,ang_d_y,ang_d_z,"

                // staged (rviz frame; ang in rad/s)
                << "lin_des0_x,lin_des0_y,lin_des0_z,"
                << "ang_des0_x,ang_des0_y,ang_des0_z,"
                << "lin_vlim_x,lin_vlim_y,lin_vlim_z,"
                << "ang_vlim_x,ang_vlim_y,ang_vlim_z,"
                << "lin_alim_x,lin_alim_y,lin_alim_z,"
                << "ang_alim_x,ang_alim_y,ang_alim_z,"
                << "lin_filt_x,lin_filt_y,lin_filt_z,"
                << "ang_filt_x,ang_filt_y,ang_filt_z,"

                // final command (controller frame; ang may be deg/s)
                << "cmd_lin_x,cmd_lin_y,cmd_lin_z,"
                << "cmd_ang_x,cmd_ang_y,cmd_ang_z,"
                
                // spike detection info
                << "spike_detected,pos_near_zero,ori_near_zero\n";

            for (const auto& s : logs_) {
                ofs << s.t << ","
                    // pose
                    << s.pos.x()  << "," << s.pos.y()  << "," << s.pos.z()  << ","
                    << s.quat.w() << "," << s.quat.x() << "," << s.quat.y() << "," << s.quat.z() << ","
                    << s.axis_angle_vec.x() << "," << s.axis_angle_vec.y() << "," << s.axis_angle_vec.z() << ","

                    // error
                    << s.pos_err.x() << "," << s.pos_err.y() << "," << s.pos_err.z() << ","
                    << s.ori_err.x() << "," << s.ori_err.y() << "," << s.ori_err.z() << ","

                    // derivatives
                    << s.pos_err_der_raw.x()  << "," << s.pos_err_der_raw.y()  << "," << s.pos_err_der_raw.z()  << ","
                    << s.ori_err_der_raw.x()  << "," << s.ori_err_der_raw.y()  << "," << s.ori_err_der_raw.z()  << ","

                    // PD terms
                    << s.lin_p.x() << "," << s.lin_p.y() << "," << s.lin_p.z() << ","
                    << s.lin_d.x() << "," << s.lin_d.y() << "," << s.lin_d.z() << ","
                    << s.ang_p.x() << "," << s.ang_p.y() << "," << s.ang_p.z() << ","
                    << s.ang_d.x() << "," << s.ang_d.y() << "," << s.ang_d.z() << ","

                    // staged
                    << s.lin_des_before_limit.x() << "," << s.lin_des_before_limit.y() << "," << s.lin_des_before_limit.z() << ","
                    << s.ang_des_before_limit.x() << "," << s.ang_des_before_limit.y() << "," << s.ang_des_before_limit.z() << ","
                    << s.lin_after_vel_limit.x()  << "," << s.lin_after_vel_limit.y()  << "," << s.lin_after_vel_limit.z()  << ","
                    << s.ang_after_vel_limit.x()  << "," << s.ang_after_vel_limit.y()  << "," << s.ang_after_vel_limit.z()  << ","
                    << s.lin_after_acc_limit.x()  << "," << s.lin_after_acc_limit.y()  << "," << s.lin_after_acc_limit.z()  << ","
                    << s.ang_after_acc_limit.x()  << "," << s.ang_after_acc_limit.y()  << "," << s.ang_after_acc_limit.z()  << ","
                    << s.lin_after_filter.x()     << "," << s.lin_after_filter.y()     << "," << s.lin_after_filter.z()     << ","
                    << s.ang_after_filter.x()     << "," << s.ang_after_filter.y()     << "," << s.ang_after_filter.z()     << ","

                    // final command (controller frame)
                    << s.lin_cmd_ctrl.x() << "," << s.lin_cmd_ctrl.y() << "," << s.lin_cmd_ctrl.z() << ","
                    << s.ang_cmd_ctrl_out.x() << "," << s.ang_cmd_ctrl_out.y() << "," << s.ang_cmd_ctrl_out.z() << ","
                    
                    // spike detection
                    << s.spike_detected << "," << s.pos_near_zero << "," << s.ori_near_zero
                    << "\n";
            }
            ofs.close();
            std::cout << "[save] Saved CSV: " << csv_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[save] Failed to save CSV: " << e.what() << std::endl;
            return;
        }

        // === 2) 调用外部 Python 绘图脚本 ===
        if (!plot_script_.empty()) {
            try {
                std::string ang_unit = publish_ang_in_degrees_ ? "deg/s" : "rad/s";
                std::string cmd = "\"" + python_interpreter_ + "\" \"" + plot_script_ + "\""
                                " --csv \"" + csv_path + "\""
                                " --outdir \"" + log_dir_ + "\""
                                " --ang-unit \"" + ang_unit + "\"";
                int ret = std::system(cmd.c_str());

                if (ret != 0) {
                    std::cerr << "[save] Plot script returned " << ret
                            << ". Check python3/pandas/matplotlib. Files at " << log_dir_ << std::endl;
                } else {
                    std::cout << "[save] Saved plots to " << log_dir_ << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "[save] Failed to exec plot script: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "[save] plot_script parameter is empty. Skipping plot.\n";
        }
    }
    
    ~VelocityControlNode()
    {
        if (control_timer_) control_timer_->cancel();
        if (gripper_timer_) gripper_timer_->cancel();
        try {
            saveLogsToCsvAndPlots();
        } catch (...) {
            // 静默
        }
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
    
    // PD控制增益
    double position_kp_, position_kd_;
    double orientation_kp_, orientation_kd_;
    
    // 速度和加速度限制
    double max_linear_vel_, max_angular_vel_;
    double max_linear_acc_, max_angular_acc_;
    
    // 滤波参数
    double filter_alpha_;                  // 速度输出滤波
    double target_filter_alpha_;          // 目标位姿滤波
    double current_filter_alpha_;         // 当前位姿滤波  
    double error_filter_alpha_;           // 误差滤波
    
    // 死区（只保留速度死区）
    double velocity_deadzone_;
    
    // 微分项尖峰检测参数
    double error_near_zero_threshold_;    // 误差接近0的阈值
    double spike_detection_factor_;       // 尖峰检测因子
    
    // Control states
    geometry_msgs::msg::PoseStamped target_pose_;
    bool target_received_ = false;
    bool joints_initialized_ = false;
    bool publish_ang_in_degrees_ = true;
    bool first_control_cycle_ = true;
    bool target_initialized_ = false;
    bool current_initialized_ = false;

    // ===== Logging =====
    bool log_enabled_ = true;
    std::string log_dir_ = "./vc_logs";
    rclcpp::Time start_time_;

    // 速度状态
    Eigen::Vector3d filtered_linear_vel_;
    Eigen::Vector3d filtered_angular_vel_;
    Eigen::Vector3d last_linear_vel_;
    Eigen::Vector3d last_angular_vel_;
    Eigen::Matrix3d R_ctrl_from_rviz_;
    
    // 滤波后的位姿和误差
    Eigen::Vector3d filtered_target_position_;
    Eigen::Quaterniond filtered_target_quaternion_;
    Eigen::Vector3d filtered_current_position_;
    Eigen::Quaterniond filtered_current_quaternion_;
    Eigen::Vector3d filtered_position_error_;
    Eigen::Vector3d filtered_orientation_error_;
    
    // PD控制相关状态（现在使用滤波后的误差）
    Eigen::Vector3d last_filtered_position_error_;
    Eigen::Vector3d last_filtered_orientation_error_;
    
    // Gripper state
    double gripper_command_ = 0.0;
    double last_gripper_command_ = -1.0;
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
    std::string python_interpreter_;
    std::string plot_script_;

    struct LogSample {
        double t;
        Eigen::Vector3d pos;
        Eigen::Quaterniond quat;
        Eigen::Vector3d axis_angle_vec;
        Eigen::Vector3d pos_err;
        Eigen::Vector3d ori_err;
        Eigen::Vector3d pos_err_der_raw;
        Eigen::Vector3d ori_err_der_raw;
        Eigen::Vector3d lin_p;
        Eigen::Vector3d lin_d;
        Eigen::Vector3d ang_p;
        Eigen::Vector3d ang_d;
        Eigen::Vector3d lin_des_before_limit;
        Eigen::Vector3d ang_des_before_limit;
        Eigen::Vector3d lin_after_vel_limit;
        Eigen::Vector3d ang_after_vel_limit;
        Eigen::Vector3d lin_after_acc_limit;
        Eigen::Vector3d ang_after_acc_limit;
        Eigen::Vector3d lin_after_filter;
        Eigen::Vector3d ang_after_filter;
        Eigen::Vector3d lin_cmd_ctrl;     
        Eigen::Vector3d ang_cmd_ctrl_out; 
        bool spike_detected;
        bool pos_near_zero;
        bool ori_near_zero;
    };

    std::vector<LogSample> logs_;
    
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
        
        // 对目标位姿进行极轻滤波
        Eigen::Vector3d new_target_position(
            msg->pose.position.x,
            msg->pose.position.y,
            msg->pose.position.z
        );
        
        Eigen::Quaterniond new_target_quaternion(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z
        );
        
        if (!target_initialized_) {
            // 第一次初始化
            filtered_target_position_ = new_target_position;
            filtered_target_quaternion_ = new_target_quaternion;
            target_initialized_ = true;
        } else {
            // 极轻滤波（快速响应）
            filtered_target_position_ = target_filter_alpha_ * new_target_position + 
                                       (1.0 - target_filter_alpha_) * filtered_target_position_;
            
            // 四元数球面插值
            filtered_target_quaternion_ = filtered_target_quaternion_.slerp(
                target_filter_alpha_, new_target_quaternion);
        }
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
        Eigen::Quaterniond qc = current_quat.normalized();
        Eigen::Quaterniond qt = target_quat.normalized();
        Eigen::Quaterniond qe = qt * qc.conjugate();
        if (qe.w() < 0.0) { qe.coeffs() *= -1.0; }

        const double w = std::clamp(qe.w(), -1.0, 1.0);
        const Eigen::Vector3d v = qe.vec();
        const double vnorm = v.norm();
        const double eps = 1e-9;

        if (vnorm < eps) {
            return 2.0 * v;
        } else {
            const double theta = 2.0 * std::atan2(vnorm, w);
            return (theta / vnorm) * v;
        }
    }

    Eigen::Vector3d applyDeadzone(const Eigen::Vector3d& value, double deadzone)
    {
        Eigen::Vector3d result;
        for (int i = 0; i < 3; ++i) {
            if (std::abs(value[i]) < deadzone) {
                result[i] = 0.0;
            } else {
                double sign = (value[i] > 0) ? 1.0 : -1.0;
                result[i] = sign * (std::abs(value[i]) - deadzone);
            }
        }
        return result;
    }
    
    // 修改为按轴独立限制
    Eigen::Vector3d limitVectorByAxis(const Eigen::Vector3d& vec, double max_value)
    {
        Eigen::Vector3d result;
        for (int i = 0; i < 3; ++i) {
            if (std::abs(vec[i]) > max_value) {
                result[i] = (vec[i] > 0) ? max_value : -max_value;
            } else {
                result[i] = vec[i];
            }
        }
        return result;
    }
    
    Eigen::Vector3d applyAccelerationLimit(const Eigen::Vector3d& desired_vel,
                                          const Eigen::Vector3d& last_vel,
                                          double max_acc)
    {
        Eigen::Vector3d vel_diff = desired_vel - last_vel;
        double max_change = max_acc * dt_;
        
        // 按轴独立限制加速度
        vel_diff = limitVectorByAxis(vel_diff, max_change);
        
        return last_vel + vel_diff;
    }
    
    Eigen::Vector3d quatToAxisAngleVec(const Eigen::Quaterniond& q_in)
    {
        Eigen::Quaterniond q = q_in.normalized();
        Eigen::AngleAxisd aa(q);
        return aa.angle() * aa.axis();
    }
    
    // 检测微分项尖峰
    bool detectDerivativeSpike(const Eigen::Vector3d& current_pos_err,
                              const Eigen::Vector3d& current_ori_err,
                              const Eigen::Vector3d& last_pos_err,
                              const Eigen::Vector3d& last_ori_err,
                              const Eigen::Vector3d& last_lin_vel,
                              const Eigen::Vector3d& last_ang_vel,
                              bool& pos_near_zero,
                              bool& ori_near_zero)
    {
        // 检查误差是否接近0
        pos_near_zero = current_pos_err.norm() < error_near_zero_threshold_;
        ori_near_zero = current_ori_err.norm() < error_near_zero_threshold_;
        
        if (!pos_near_zero && !ori_near_zero) {
            return false; // 误差都不接近0，不需要检测
        }
        
        // 计算误差变化量
        Eigen::Vector3d pos_err_diff = current_pos_err - last_pos_err;
        Eigen::Vector3d ori_err_diff = current_ori_err - last_ori_err;
        
        // 计算检测阈值
        double pos_threshold = spike_detection_factor_ * dt_ * last_lin_vel.norm();
        double ori_threshold = spike_detection_factor_ * dt_ * last_ang_vel.norm();
        
        bool pos_spike = pos_near_zero && (pos_err_diff.norm() > pos_threshold);
        bool ori_spike = ori_near_zero && (ori_err_diff.norm() > ori_threshold);
        
        return pos_spike || ori_spike;
    }
    
    void controlLoop()
    {
        if (!target_received_ || !joints_initialized_ || !target_initialized_) {
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Waiting for target pose and joint states...");
            return;
        }
        
        // Get current pose from FK
        auto current_pose = kdlFrameToPoseMsg(current_ee_frame_);
        
        // ========== 对当前位姿进行轻滤波 ==========
        Eigen::Vector3d raw_current_position(
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            current_pose.pose.position.z
        );
        
        Eigen::Quaterniond raw_current_quaternion(
            current_pose.pose.orientation.w,
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z
        );
        
        if (!current_initialized_) {
            // 第一次初始化
            filtered_current_position_ = raw_current_position;
            filtered_current_quaternion_ = raw_current_quaternion;
            current_initialized_ = true;
        } else {
            // 轻滤波（减少测量噪音）
            filtered_current_position_ = current_filter_alpha_ * raw_current_position + 
                                        (1.0 - current_filter_alpha_) * filtered_current_position_;
            
            // 四元数球面插值
            filtered_current_quaternion_ = filtered_current_quaternion_.slerp(
                current_filter_alpha_, raw_current_quaternion);
        }
        
        // ========== 计算原始误差 ==========
        Eigen::Vector3d raw_position_error = filtered_target_position_ - filtered_current_position_;
        Eigen::Vector3d raw_orientation_error = computeOrientationError(
            filtered_current_quaternion_, filtered_target_quaternion_);
        
        // 移除位置和角度死区处理
        
        // ========== 滤波误差（用于比例项和微分项） ==========
        if (first_control_cycle_) {
            filtered_position_error_ = raw_position_error;
            filtered_orientation_error_ = raw_orientation_error;
            last_filtered_position_error_ = raw_position_error;
            last_filtered_orientation_error_ = raw_orientation_error;
        } else {
            filtered_position_error_ = error_filter_alpha_ * raw_position_error + 
                                      (1.0 - error_filter_alpha_) * filtered_position_error_;
            filtered_orientation_error_ = error_filter_alpha_ * raw_orientation_error + 
                                         (1.0 - error_filter_alpha_) * filtered_orientation_error_;
        }
        
        // ========== 计算微分项（不除以dt，直接用差值） ==========
        Eigen::Vector3d position_derivative_raw, orientation_derivative_raw;
        bool spike_detected = false, pos_near_zero = false, ori_near_zero = false;
        
        if (first_control_cycle_) {
            position_derivative_raw.setZero();
            orientation_derivative_raw.setZero();
            first_control_cycle_ = false;
        } else {
            // 检测微分项尖峰
            spike_detected = detectDerivativeSpike(
                filtered_position_error_, filtered_orientation_error_,
                last_filtered_position_error_, last_filtered_orientation_error_,
                last_linear_vel_, last_angular_vel_,
                pos_near_zero, ori_near_zero);
            
            if (spike_detected) {
                // 检测到尖峰，将微分项设为0
                position_derivative_raw.setZero();
                orientation_derivative_raw.setZero();
                RCLCPP_DEBUG(this->get_logger(), "Derivative spike detected, setting derivatives to zero");
            } else {
                // 正常计算微分项（不除以dt）
                position_derivative_raw = filtered_position_error_ - last_filtered_position_error_;
                orientation_derivative_raw = filtered_orientation_error_ - last_filtered_orientation_error_;
            }
        }
        
        // 保存当前滤波后的误差供下次使用
        last_filtered_position_error_ = filtered_position_error_;
        last_filtered_orientation_error_ = filtered_orientation_error_;
        
        // ========== PD控制律 ==========
        // 比例项和微分项都使用滤波后的误差
        Eigen::Vector3d lin_p = position_kp_ * filtered_position_error_;
        Eigen::Vector3d lin_d = position_kd_ * position_derivative_raw;  // 不除以dt
        Eigen::Vector3d ang_p = orientation_kp_ * filtered_orientation_error_;
        Eigen::Vector3d ang_d = orientation_kd_ * orientation_derivative_raw;  // 不除以dt

        Eigen::Vector3d desired_linear_vel = lin_p + lin_d;
        Eigen::Vector3d desired_angular_vel = ang_p + ang_d;
        
        // 记录
        Eigen::Vector3d lin_des_before_limit = desired_linear_vel;
        Eigen::Vector3d ang_des_before_limit = desired_angular_vel;
        
        // ========== 应用速度限制（按轴独立） ==========
        desired_linear_vel = limitVectorByAxis(desired_linear_vel, max_linear_vel_);
        desired_angular_vel = limitVectorByAxis(desired_angular_vel, max_angular_vel_);
        
        Eigen::Vector3d lin_after_vel_limit = desired_linear_vel;
        Eigen::Vector3d ang_after_vel_limit = desired_angular_vel;
        
        // ========== 应用加速度限制（按轴独立） ==========
        desired_linear_vel = applyAccelerationLimit(desired_linear_vel, last_linear_vel_, max_linear_acc_);
        desired_angular_vel = applyAccelerationLimit(desired_angular_vel, last_angular_vel_, max_angular_acc_);
        
        Eigen::Vector3d lin_after_acc_limit = desired_linear_vel;
        Eigen::Vector3d ang_after_acc_limit = desired_angular_vel;
        
        // ========== 应用输出滤波 ==========
        filtered_linear_vel_ = filter_alpha_ * desired_linear_vel + 
                              (1.0 - filter_alpha_) * filtered_linear_vel_;
        filtered_angular_vel_ = filter_alpha_ * desired_angular_vel + 
                               (1.0 - filter_alpha_) * filtered_angular_vel_;
        
        Eigen::Vector3d lin_after_filter = filtered_linear_vel_;
        Eigen::Vector3d ang_after_filter = filtered_angular_vel_;
        
        // 应用速度死区（只对最终输出速度）
        filtered_linear_vel_ = applyDeadzone(filtered_linear_vel_, velocity_deadzone_);
        filtered_angular_vel_ = applyDeadzone(filtered_angular_vel_, velocity_deadzone_);
        
        // ========== 坐标变换 ==========
        Eigen::Vector3d lin_rviz = filtered_linear_vel_;
        Eigen::Vector3d ang_rviz = filtered_angular_vel_;

        Eigen::Vector3d lin_ctrl = R_ctrl_from_rviz_ * lin_rviz;
        Eigen::Vector3d ang_ctrl = R_ctrl_from_rviz_ * ang_rviz;

        Eigen::Vector3d ang_ctrl_out = ang_ctrl;
        if (publish_ang_in_degrees_) {
            constexpr double RAD2DEG = 180.0 / M_PI;
            ang_ctrl_out *= RAD2DEG;
        }

        // Update last velocities
        last_linear_vel_ = filtered_linear_vel_;
        last_angular_vel_ = filtered_angular_vel_;
        
        // ========== 发布控制命令 ==========
        geometry_msgs::msg::Twist twist_msg;
        twist_msg.linear.x  = lin_ctrl.x();
        twist_msg.linear.y  = lin_ctrl.y();
        twist_msg.linear.z  = lin_ctrl.z();
        twist_msg.angular.x = ang_ctrl_out.x();
        twist_msg.angular.y = ang_ctrl_out.y();
        twist_msg.angular.z = ang_ctrl_out.z();
        
        twist_pub_->publish(twist_msg);
        
        // ========== 记录日志 ==========
        if (log_enabled_) {
            LogSample s;
            s.t = (this->now() - start_time_).seconds();

            s.pos = filtered_current_position_;
            s.quat = filtered_current_quaternion_;
            s.axis_angle_vec = quatToAxisAngleVec(s.quat);

            s.pos_err = filtered_position_error_;     // 使用滤波后的误差记录
            s.ori_err = filtered_orientation_error_;

            s.pos_err_der_raw  = position_derivative_raw;
            s.ori_err_der_raw  = orientation_derivative_raw;

            s.lin_p = lin_p;
            s.lin_d = lin_d;
            s.ang_p = ang_p;
            s.ang_d = ang_d;

            s.lin_des_before_limit = lin_des_before_limit;
            s.ang_des_before_limit = ang_des_before_limit;
            s.lin_after_vel_limit  = lin_after_vel_limit;
            s.ang_after_vel_limit  = ang_after_vel_limit;
            s.lin_after_acc_limit  = lin_after_acc_limit;
            s.ang_after_acc_limit  = ang_after_acc_limit;
            s.lin_after_filter     = lin_after_filter;
            s.ang_after_filter     = ang_after_filter;

            s.lin_cmd_ctrl     = lin_ctrl;
            s.ang_cmd_ctrl_out = ang_ctrl_out;
            
            s.spike_detected = spike_detected;
            s.pos_near_zero = pos_near_zero;
            s.ori_near_zero = ori_near_zero;

            logs_.push_back(std::move(s));
        }
        
        // ========== 调试输出 ==========
        static int counter = 0;
        if (++counter % static_cast<int>(control_frequency_) == 0) {
            if (spike_detected) {
                RCLCPP_WARN(this->get_logger(), "Derivative spike detected and suppressed");
            }
            
            RCLCPP_INFO(this->get_logger(), 
                       "Filtered Err | Pos: [%.3f, %.3f, %.3f]m | Orient: [%.3f, %.3f, %.3f]rad",
                       filtered_position_error_.x(), filtered_position_error_.y(), filtered_position_error_.z(),
                       filtered_orientation_error_.x(), filtered_orientation_error_.y(), filtered_orientation_error_.z());
            
            RCLCPP_INFO(this->get_logger(), 
                       "Derivatives | Pos: [%.3f, %.3f, %.3f] | Orient: [%.3f, %.3f, %.3f] (no dt division)",
                       position_derivative_raw.x(), position_derivative_raw.y(), position_derivative_raw.z(),
                       orientation_derivative_raw.x(), orientation_derivative_raw.y(), orientation_derivative_raw.z());
            
            RCLCPP_INFO(this->get_logger(),
                       "Output vel | Linear: [%.3f, %.3f, %.3f]m/s | Angular: [%.3f, %.3f, %.3f]%s",
                       lin_ctrl.x(), lin_ctrl.y(), lin_ctrl.z(),
                       ang_ctrl_out.x(), ang_ctrl_out.y(), ang_ctrl_out.z(),
                       publish_ang_in_degrees_ ? "deg/s" : "rad/s");
        }
    }
    
    void gripperControlLoop()
    {
        if (std::abs(gripper_command_ - last_gripper_command_) < 0.01) {
            return;
        }

        if (!gripper_action_client_->action_server_is_ready()) {
            RCLCPP_DEBUG(this->get_logger(), "Gripper action server not ready");
            return;
        }

        auto goal_msg = GripperCommand::Goal();
        goal_msg.command.position = gripper_command_;
        goal_msg.command.max_effort = 10.0;

        if (current_gripper_goal_ &&
            current_gripper_goal_->get_status() == rclcpp_action::GoalStatus::STATUS_EXECUTING) {
            (void)gripper_action_client_->async_cancel_goal(current_gripper_goal_);
        }

        RCLCPP_DEBUG(this->get_logger(), "Sending gripper command: %.3f", gripper_command_);
        (void)gripper_action_client_->async_send_goal(goal_msg, send_goal_options_);

        last_gripper_command_ = gripper_command_;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<VelocityControlNode>();

    rclcpp::on_shutdown([node](){
        node->saveLogsToCsvAndPlots();
    });

    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}