from setuptools import find_packages, setup

package_name = 'xr_interface_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ming',
    maintainer_email='googoo000078@gmail.com',
    description='ROS2 service node for XR device interface',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'xr_interface_server = xr_interface_node.xr_interface_server:main',
        ],
    },
)
