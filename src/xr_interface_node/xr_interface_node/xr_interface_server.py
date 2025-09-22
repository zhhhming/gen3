#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from xr_interfaces.srv import GetPose, GetKeyValue, GetButtonState
from xr_interfaces.msg import KeyValue, Pose
import numpy as np
import threading

#xrtsdk安装在虚拟环境里，然后ros在自己的python里运行
#编译时先是interface再是node，interface里有自定义信息。
import sys

# 把 xrobotoolkit_sdk 的 egg 路径直接加到 sys.path， 写死的，依据虚拟环境里xrtsdk的具体位置
sys.path.insert(0, "/home/ming/miniconda3/envs/xrrobotics/lib/python3.10/site-packages/xrobotoolkit_sdk-1.0.2-py3.10-linux-x86_64.egg")

# 现在可以导入
import xrobotoolkit_sdk as xrt
print("Successfully imported xrobotoolkit_sdk from:", xrt.__file__)


class XRInterfaceNode(Node):
    """ROS2 Service node for XR device interface"""

    def __init__(self):
        super().__init__('xr_interface_node')
        
        # Initialize XR SDK
        self._init_xr_sdk()
        
        # Create service servers (保留button state服务)
        self.pose_service = self.create_service(
            GetPose, 
            'xr/get_pose', 
            self.get_pose_callback
        )
        
        self.key_value_service = self.create_service(
            GetKeyValue,
            'xr/get_key_value',
            self.get_key_value_callback
        )
        
        self.button_state_service = self.create_service(
            GetButtonState,
            'xr/get_button_state',
            self.get_button_state_callback
        )
        
        # Create publishers for high-frequency data
        self.right_grip_pub = self.create_publisher(
            KeyValue, 
            'xr/right_grip', 
            10
        )
        
        self.right_trigger_pub = self.create_publisher(
            KeyValue, 
            'xr/right_trigger', 
            10
        )
        
        self.right_controller_pose_pub = self.create_publisher(
            Pose, 
            'xr/right_controller_pose', 
            10
        )
        
        # Create timer for 500Hz publishing (0.002 seconds = 2ms)
        self.publish_timer = self.create_timer(0.004, self.publish_high_freq_data)
        
        # Thread lock for SDK access
        self.sdk_lock = threading.Lock()
        
        self.get_logger().info('XR Interface Node initialized successfully')
        self.get_logger().info('Services available:')
        self.get_logger().info('  - /xr/get_pose')
        self.get_logger().info('  - /xr/get_key_value')
        self.get_logger().info('  - /xr/get_button_state')
        self.get_logger().info('Topics publishing at 500Hz:')
        self.get_logger().info('  - /xr/right_grip')
        self.get_logger().info('  - /xr/right_trigger')
        self.get_logger().info('  - /xr/right_controller_pose')

    def _init_xr_sdk(self):
        """Initialize the XR SDK"""
        try:
            xrt.init()
            self.get_logger().info('XRoboToolkit SDK initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize XR SDK: {str(e)}')
            raise

    def publish_high_freq_data(self):
        """Publish high-frequency data at 500Hz"""
        with self.sdk_lock:
            try:
                # Publish right grip value
                right_grip_msg = KeyValue()
                right_grip_msg.name = "right_grip"
                right_grip_msg.value = float(xrt.get_right_grip())
                self.right_grip_pub.publish(right_grip_msg)
                
                # Publish right trigger value
                right_trigger_msg = KeyValue()
                right_trigger_msg.name = "right_trigger"
                right_trigger_msg.value = float(xrt.get_right_trigger())
                self.right_trigger_pub.publish(right_trigger_msg)
                
                # Publish right controller pose
                right_controller_pose_msg = Pose()
                right_controller_pose_msg.name = "right_controller"
                pose_array = xrt.get_right_controller_pose()
                right_controller_pose_msg.pose = pose_array.tolist() if isinstance(pose_array, np.ndarray) else list(pose_array)
                self.right_controller_pose_pub.publish(right_controller_pose_msg)
                
            except Exception as e:
                self.get_logger().error(f"Error publishing high-frequency data: {str(e)}")

    def get_pose_callback(self, request, response):
        """Handle get_pose service requests"""
        with self.sdk_lock:
            try:
                valid_names = ["left_controller", "right_controller", "headset"]
                
                if request.name not in valid_names:
                    response.success = False
                    response.error_message = f"Invalid name: {request.name}. Valid names: {valid_names}"
                    self.get_logger().warning(f"Invalid pose request: {request.name}")
                    return response
                
                # Get pose from SDK
                if request.name == "left_controller":
                    pose_array = xrt.get_left_controller_pose()
                elif request.name == "right_controller":
                    pose_array = xrt.get_right_controller_pose()
                elif request.name == "headset":
                    pose_array = xrt.get_headset_pose()
                
                # Convert numpy array to list and assign to response
                response.pose = pose_array.tolist() if isinstance(pose_array, np.ndarray) else list(pose_array)
                response.success = True
                response.error_message = ""
                
                self.get_logger().debug(f"Pose request for {request.name} successful")
                
            except Exception as e:
                response.success = False
                response.error_message = str(e)
                response.pose = []
                self.get_logger().error(f"Error getting pose for {request.name}: {str(e)}")
        
        return response

    def get_key_value_callback(self, request, response):
        """Handle get_key_value service requests"""
        with self.sdk_lock:
            try:
                valid_names = ["left_trigger", "right_trigger", "left_grip", "right_grip"]
                
                if request.name not in valid_names:
                    response.success = False
                    response.error_message = f"Invalid name: {request.name}. Valid names: {valid_names}"
                    response.value = 0.0
                    self.get_logger().warning(f"Invalid key value request: {request.name}")
                    return response
                
                # Get key value from SDK
                if request.name == "left_trigger":
                    value = xrt.get_left_trigger()
                elif request.name == "right_trigger":
                    value = xrt.get_right_trigger()
                elif request.name == "left_grip":
                    value = xrt.get_left_grip()
                elif request.name == "right_grip":
                    value = xrt.get_right_grip()
                
                response.value = float(value)
                response.success = True
                response.error_message = ""
                
                self.get_logger().debug(f"Key value request for {request.name}: {response.value:.3f}")
                
            except Exception as e:
                response.success = False
                response.error_message = str(e)
                response.value = 0.0
                self.get_logger().error(f"Error getting key value for {request.name}: {str(e)}")
        
        return response

    def get_button_state_callback(self, request, response):
        """Handle get_button_state service requests"""
        with self.sdk_lock:
            try:
                valid_names = [
                    "A", "B", "X", "Y",
                    "left_menu_button", "right_menu_button",
                    "left_axis_click", "right_axis_click"
                ]
                
                if request.name not in valid_names:
                    response.success = False
                    response.error_message = f"Invalid name: {request.name}. Valid names: {valid_names}"
                    response.pressed = False
                    self.get_logger().warning(f"Invalid button state request: {request.name}")
                    return response
                
                # Get button state from SDK
                if request.name == "A":
                    pressed = xrt.get_A_button()
                elif request.name == "B":
                    pressed = xrt.get_B_button()
                elif request.name == "X":
                    pressed = xrt.get_X_button()
                elif request.name == "Y":
                    pressed = xrt.get_Y_button()
                elif request.name == "left_menu_button":
                    pressed = xrt.get_left_menu_button()
                elif request.name == "right_menu_button":
                    pressed = xrt.get_right_menu_button()
                elif request.name == "left_axis_click":
                    pressed = xrt.get_left_axis_click()
                elif request.name == "right_axis_click":
                    pressed = xrt.get_right_axis_click()
                
                response.pressed = bool(pressed)
                response.success = True
                response.error_message = ""
                
                self.get_logger().debug(f"Button state request for {request.name}: {response.pressed}")
                
            except Exception as e:
                response.success = False
                response.error_message = str(e)
                response.pressed = False
                self.get_logger().error(f"Error getting button state for {request.name}: {str(e)}")
        
        return response

    def destroy_node(self):
        """Clean up when node is destroyed"""
        try:
            with self.sdk_lock:
                xrt.close()
            self.get_logger().info('XR SDK closed successfully')
        except Exception as e:
            self.get_logger().error(f'Error closing XR SDK: {str(e)}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        xr_interface_node = XRInterfaceNode()
        rclpy.spin(xr_interface_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in XR Interface Node: {e}")
    finally:
        if 'xr_interface_node' in locals():
            xr_interface_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()