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
        this->declare_parameter("position_kp", 3.0);      // 位置比例增益（稍微提高）
        this->declare_parameter("position_kd", 2.0);      // 位置微分增益（新增）
        this->declare_parameter("orientation_kp", 3.0);   // 姿态比例增益（稍微提高）
        this->declare_parameter("orientation_kd", 2.0);   // 姿态微分增益（新增）
        
        // 速度和加速度限制（调整）
        this->declare_parameter("max_linear_velocity", 0.5);      // 降低最大线速度以提高稳定性
        this->declare_parameter("max_angular_velocity", 1.0);     // 降低最大角速度
        this->declare_parameter("max_linear_acceleration", 2.0);  // 降低加速度限制以更平滑
        this->declare_parameter("max_angular_acceleration", 5.0); // 降低角加速度
        
        // 滤波参数
        this->declare_parameter("velocity_filter_alpha", 0.5);    // 降低滤波系数以更平滑
        this->declare_parameter("derivative_filter_alpha", 0.5);  // 微分项滤波（新增）
        
        // 死区参数
        this->declare_parameter("deadzone", 0.002);               // 2mm位置死区
        this->declare_parameter("angular_deadzone", 0.015);        // ~0.57度姿态死区
        this->declare_parameter("velocity_deadzone", 0.003);      // 速度死区（新增）
        
        this->declare_parameter("publish_angular_in_degrees", true);
        this->declare_parameter("log_enabled", true);
        this->declare_parameter("log_dir", std::string("./vc_logs"));
        this->declare_parameter("plot_script", std::string("/home/ming/ros2_ws/vc_logs/plot_vc_log.py"));  // 为空则不画
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
        
        // 滤波参数
        filter_alpha_ = this->get_parameter("velocity_filter_alpha").as_double();
        derivative_filter_alpha_ = this->get_parameter("derivative_filter_alpha").as_double();
        
        // 死区
        position_deadzone_ = this->get_parameter("deadzone").as_double();
        angular_deadzone_ = this->get_parameter("angular_deadzone").as_double();
        velocity_deadzone_ = this->get_parameter("velocity_deadzone").as_double();
        
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
        
        // 初始化PD控制相关变量
        last_position_error_.setZero();
        last_orientation_error_.setZero();
        filtered_position_derivative_.setZero();
        filtered_orientation_derivative_.setZero();
        first_control_cycle_ = true;
        
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
        RCLCPP_INFO(this->get_logger(), "PD Velocity Control Node initialized");
        RCLCPP_INFO(this->get_logger(), "Position: Kp=%.2f, Kd=%.2f | Orientation: Kp=%.2f, Kd=%.2f", 
                   position_kp_, position_kd_, orientation_kp_, orientation_kd_);
        RCLCPP_INFO(this->get_logger(), "Max linear vel: %.2f m/s, Max angular vel: %.2f rad/s", 
                   max_linear_vel_, max_angular_vel_);
        RCLCPP_INFO(this->get_logger(), "Max linear acc: %.2f m/s², Max angular acc: %.2f rad/s²", 
                   max_linear_acc_, max_angular_acc_);
        RCLCPP_INFO(this->get_logger(), "Control frequency: %.1f Hz, Gripper frequency: %.1f Hz", 
                   control_frequency_, gripper_control_frequency_);
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

                // derivatives (raw & filt)
                << "der_pos_raw_x,der_pos_raw_y,der_pos_raw_z,"
                << "der_pos_filt_x,der_pos_filt_y,der_pos_filt_z,"
                << "der_ori_raw_x,der_ori_raw_y,der_ori_raw_z,"
                << "der_ori_filt_x,der_ori_filt_y,der_ori_filt_z,"

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
                << "cmd_ang_x,cmd_ang_y,cmd_ang_z\n";

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
                    << s.pos_err_der_filt.x() << "," << s.pos_err_der_filt.y() << "," << s.pos_err_der_filt.z() << ","
                    << s.ori_err_der_raw.x()  << "," << s.ori_err_der_raw.y()  << "," << s.ori_err_der_raw.z()  << ","
                    << s.ori_err_der_filt.x() << "," << s.ori_err_der_filt.y() << "," << s.ori_err_der_filt.z() << ","

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
                    << s.ang_cmd_ctrl_out.x() << "," << s.ang_cmd_ctrl_out.y() << "," << s.ang_cmd_ctrl_out.z()
                    << "\n";
            }
            ofs.close();
            std::cout << "[save] Saved CSV: " << csv_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[save] Failed to save CSV: " << e.what() << std::endl;
            return;
        }

        // === 2) 调用外部 Python 绘图脚本（如果提供了路径） ===
        if (!plot_script_.empty()) {
            try {
                // 角速度单位标签：与你的 publish_ang_in_degrees_ 一致
                std::string ang_unit = publish_ang_in_degrees_ ? "deg/s" : "rad/s";

                // 例如：python3 /path/to/plot_vc_log.py --csv "<csv_path>" --outdir "<log_dir_>" --ang-unit "deg/s"
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
                saveLogsToCsvAndPlots();  // 仅本地文件操作
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
    double filter_alpha_;
    double derivative_filter_alpha_;
    
    // 死区
    double position_deadzone_, angular_deadzone_;
    double velocity_deadzone_;
    
    // Control states
    geometry_msgs::msg::PoseStamped target_pose_;
    bool target_received_ = false;
    bool joints_initialized_ = false;
    bool publish_ang_in_degrees_ = true;
    bool first_control_cycle_ = true;

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
    
    // PD控制相关状态
    Eigen::Vector3d last_position_error_;
    Eigen::Vector3d last_orientation_error_;
    Eigen::Vector3d filtered_position_derivative_;
    Eigen::Vector3d filtered_orientation_derivative_;
    
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

        // 当前末端位姿（base_link）
        Eigen::Vector3d pos;
        Eigen::Quaterniond quat;
        Eigen::Vector3d axis_angle_vec;

        // 误差（rviz坐标系）
        Eigen::Vector3d pos_err; // m
        Eigen::Vector3d ori_err; // rad (角轴小角向量)

        // 误差导数（原始/滤波后）
        Eigen::Vector3d pos_err_der_raw;   // m/s
        Eigen::Vector3d pos_err_der_filt;  // m/s
        Eigen::Vector3d ori_err_der_raw;   // rad/s
        Eigen::Vector3d ori_err_der_filt;  // rad/s

        // PD 分量（rviz坐标系）
        Eigen::Vector3d lin_p; // = Kp*pos_err (m/s)
        Eigen::Vector3d lin_d; // = Kd*pos_err_der_filt (m/s)
        Eigen::Vector3d ang_p; // = Kp*ori_err (rad/s)
        Eigen::Vector3d ang_d; // = Kd*ori_err_der_filt (rad/s)

        // 速度指令分阶段（rviz坐标系；角速度均为 rad/s）
        Eigen::Vector3d lin_des_before_limit;
        Eigen::Vector3d ang_des_before_limit;
        Eigen::Vector3d lin_after_vel_limit;
        Eigen::Vector3d ang_after_vel_limit;
        Eigen::Vector3d lin_after_acc_limit;
        Eigen::Vector3d ang_after_acc_limit;
        Eigen::Vector3d lin_after_filter;
        Eigen::Vector3d ang_after_filter;

        // 最终下发（控制器坐标系；角速度可能是 deg/s，取决于 publish_ang_in_degrees_）
        Eigen::Vector3d lin_cmd_ctrl;     
        Eigen::Vector3d ang_cmd_ctrl_out; 
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
        Eigen::Quaterniond qe = qt * qc.conjugate();     // 误差: 从 current 旋到 target
        if (qe.w() < 0.0) { qe.coeffs() *= -1.0; }       // 取最短路径（不改变实际旋转）

        // log(q) = (theta * axis), 其中 theta = 2*atan2(||v||, w), v=vec(q)
        const double w = std::clamp(qe.w(), -1.0, 1.0);
        const Eigen::Vector3d v = qe.vec();
        const double vnorm = v.norm();
        const double eps = 1e-9;

        if (vnorm < eps) {
            // 小角度极限：theta*axis ≈ 2*v
            return 2.0 * v;
        } else {
            const double theta = 2.0 * std::atan2(vnorm, w); // ∈ (0, π]
            return (theta / vnorm) * v;  // = angle * axis
        }
    }

    Eigen::Vector3d applyDeadzone(const Eigen::Vector3d& value, double deadzone)
    {
        Eigen::Vector3d result;
        for (int i = 0; i < 3; ++i) {
            if (std::abs(value[i]) < deadzone) {
                result[i] = 0.0;
            } else {
                // 可选：从死区边缘开始计算（避免突变）
                double sign = (value[i] > 0) ? 1.0 : -1.0;
                result[i] = sign * (std::abs(value[i]) - deadzone);
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
    Eigen::Vector3d quatToAxisAngleVec(const Eigen::Quaterniond& q_in)
    {
        // 归一化，构造 AngleAxis
        Eigen::Quaterniond q = q_in.normalized();
        Eigen::AngleAxisd aa(q);
        // 轴向量*角度（弧度），这样便于画三个通道
        return aa.angle() * aa.axis();
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
        
        // ========== 计算位置误差 ==========
        Eigen::Vector3d position_error;
        position_error << target_pose_.pose.position.x - current_pose.pose.position.x,
                         target_pose_.pose.position.y - current_pose.pose.position.y,
                         target_pose_.pose.position.z - current_pose.pose.position.z;
        //目标和当前位置未滤波
        
        // Apply deadzone to position error
        position_error = applyDeadzone(position_error, position_deadzone_);
        
        // ========== 计算姿态误差 ==========
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
        
        //目标和当前位姿未滤波

        Eigen::Vector3d orientation_error = computeOrientationError(current_quat, target_quat);
        orientation_error = applyDeadzone(orientation_error, angular_deadzone_);
        
        // ========== 计算误差导数（微分项） ==========
        Eigen::Vector3d position_derivative_raw, orientation_derivative_raw;
        
        if (first_control_cycle_) {
            // 第一个控制周期，导数设为0
            position_derivative_raw.setZero();
            orientation_derivative_raw.setZero();
            filtered_position_derivative_.setZero();
            filtered_orientation_derivative_.setZero();
            first_control_cycle_ = false;
        } else {
                position_derivative_raw    = (position_error - last_position_error_) / dt_;
                orientation_derivative_raw = (orientation_error - last_orientation_error_) / dt_;

                filtered_position_derivative_   = derivative_filter_alpha_ * position_derivative_raw
                                                + (1.0 - derivative_filter_alpha_) * filtered_position_derivative_;
                filtered_orientation_derivative_ = derivative_filter_alpha_ * orientation_derivative_raw
                                                + (1.0 - derivative_filter_alpha_) * filtered_orientation_derivative_;
        }
        Eigen::Vector3d position_derivative   = filtered_position_derivative_;
        Eigen::Vector3d orientation_derivative = filtered_orientation_derivative_;
        // 保存当前误差供下次使用
        last_position_error_ = position_error;
        last_orientation_error_ = orientation_error;
        
        // ========== PD控制律 ==========
        // u = Kp * e + Kd * de/dt
        //kp的保存，kd的保存
        Eigen::Vector3d lin_p = position_kp_   * position_error;
        Eigen::Vector3d lin_d = position_kd_   * position_derivative;
        Eigen::Vector3d ang_p = orientation_kp_* orientation_error;       // rad/s
        Eigen::Vector3d ang_d = orientation_kd_* orientation_derivative;  // rad/s

        Eigen::Vector3d desired_linear_vel = lin_p + lin_d;
        
        Eigen::Vector3d desired_angular_vel = ang_p + ang_d;
        //用于记录
        Eigen::Vector3d lin_des_before_limit = desired_linear_vel;
        Eigen::Vector3d ang_des_before_limit = desired_angular_vel;
        // ========== 应用速度限制 ==========
        desired_linear_vel = limitVector(desired_linear_vel, max_linear_vel_);
        desired_angular_vel = limitVector(desired_angular_vel, max_angular_vel_);
        //速度限制后的保存
        //用于记录
        Eigen::Vector3d lin_after_vel_limit = desired_linear_vel;
        Eigen::Vector3d ang_after_vel_limit = desired_angular_vel;
        
        // ========== 应用加速度限制 ==========
        desired_linear_vel = applyAccelerationLimit(desired_linear_vel, last_linear_vel_, max_linear_acc_);
        desired_angular_vel = applyAccelerationLimit(desired_angular_vel, last_angular_vel_, max_angular_acc_);
        //加速度限制后的保存
        //用于记录
        Eigen::Vector3d lin_after_acc_limit = desired_linear_vel;
        Eigen::Vector3d ang_after_acc_limit = desired_angular_vel;
        // ========== 应用输出滤波 ==========
        filtered_linear_vel_ = filter_alpha_ * desired_linear_vel + 
                              (1.0 - filter_alpha_) * filtered_linear_vel_;
        filtered_angular_vel_ = filter_alpha_ * desired_angular_vel + 
                               (1.0 - filter_alpha_) * filtered_angular_vel_;
        //滤波的保存
        //用于记录
        Eigen::Vector3d lin_after_filter = filtered_linear_vel_;
        Eigen::Vector3d ang_after_filter = filtered_angular_vel_;
        // 应用速度死区（避免微小抖动）
        filtered_linear_vel_ = applyDeadzone(filtered_linear_vel_, velocity_deadzone_);
        filtered_angular_vel_ = applyDeadzone(filtered_angular_vel_, velocity_deadzone_);
        
        // ========== 坐标变换 ==========
        Eigen::Vector3d lin_rviz = filtered_linear_vel_;
        Eigen::Vector3d ang_rviz = filtered_angular_vel_;

        // 用置换矩阵映射到控制器坐标
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
        if (log_enabled_) {
            LogSample s;
            s.t = (this->now() - start_time_).seconds();

            // 当前末端位姿：用你已有的 current_pose
            s.pos = Eigen::Vector3d(
                current_pose.pose.position.x,
                current_pose.pose.position.y,
                current_pose.pose.position.z
            );
            s.quat = Eigen::Quaterniond(
                current_pose.pose.orientation.w,
                current_pose.pose.orientation.x,
                current_pose.pose.orientation.y,
                current_pose.pose.orientation.z
            );
            s.axis_angle_vec = quatToAxisAngleVec(s.quat);

            // 误差（已经算好的）
            s.pos_err = position_error;        // m
            s.ori_err = orientation_error;     // rad

            // 导数（原始/滤波后）
            s.pos_err_der_raw  = position_derivative_raw;
            s.pos_err_der_filt = filtered_position_derivative_;
            s.ori_err_der_raw  = orientation_derivative_raw;
            s.ori_err_der_filt = filtered_orientation_derivative_;

                // PD 分量
            s.lin_p = lin_p;
            s.lin_d = lin_d;
            s.ang_p = ang_p;
            s.ang_d = ang_d;

                // 分阶段（rviz系，角速度单位：rad/s）
            s.lin_des_before_limit = lin_des_before_limit;
            s.ang_des_before_limit = ang_des_before_limit;
            s.lin_after_vel_limit  = lin_after_vel_limit;
            s.ang_after_vel_limit  = ang_after_vel_limit;
            s.lin_after_acc_limit  = lin_after_acc_limit;
            s.ang_after_acc_limit  = ang_after_acc_limit;
            s.lin_after_filter     = lin_after_filter;
            s.ang_after_filter     = ang_after_filter;

            // 速度指令（控制器坐标，已映射 & 角速度可能已转为deg/s）
            s.lin_cmd_ctrl     = lin_ctrl;     // m/s
            s.ang_cmd_ctrl_out = ang_ctrl_out; // rad/s or deg/s (依据配置)

            logs_.push_back(std::move(s));
        }
        // ========== 调试输出 ==========
        static int counter = 0;
        if (++counter % static_cast<int>(control_frequency_) == 0) {  // 每秒一次
            RCLCPP_INFO(this->get_logger(), 
                       "PD Control | Pos err: [%.3f, %.3f, %.3f]m | Orient err: [%.3f, %.3f, %.3f]rad",
                       position_error.x(), position_error.y(), position_error.z(),
                       orientation_error.x(), orientation_error.y(), orientation_error.z());
            
            RCLCPP_INFO(this->get_logger(), 
                       "Derivatives | Pos: [%.3f, %.3f, %.3f]m/s | Orient: [%.3f, %.3f, %.3f]rad/s",
                       position_derivative.x(), position_derivative.y(), position_derivative.z(),
                       orientation_derivative.x(), orientation_derivative.y(), orientation_derivative.z());
            
            RCLCPP_INFO(this->get_logger(),
                       "Output vel | Linear: [%.3f, %.3f, %.3f]m/s | Angular: [%.3f, %.3f, %.3f]%s",
                       lin_ctrl.x(), lin_ctrl.y(), lin_ctrl.z(),
                       ang_ctrl_out.x(), ang_ctrl_out.y(), ang_ctrl_out.z(),
                       publish_ang_in_degrees_ ? "deg/s" : "rad/s");
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

        auto goal_msg = GripperCommand::Goal();
        goal_msg.command.position = gripper_command_;
        goal_msg.command.max_effort = 10.0;

        // 若上个goal仍在执行，发起取消（非阻塞）
        if (current_gripper_goal_ &&
            current_gripper_goal_->get_status() == rclcpp_action::GoalStatus::STATUS_EXECUTING) {
            (void)gripper_action_client_->async_cancel_goal(current_gripper_goal_);
        }

        RCLCPP_DEBUG(this->get_logger(), "Sending gripper command: %.3f", gripper_command_);

        // 直接发送
        (void)gripper_action_client_->async_send_goal(goal_msg, send_goal_options_);

        last_gripper_command_ = gripper_command_;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<VelocityControlNode>();

    // 在全局 shutdown 时调用保存（仅做本地 IO，不做 ROS 调用）
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