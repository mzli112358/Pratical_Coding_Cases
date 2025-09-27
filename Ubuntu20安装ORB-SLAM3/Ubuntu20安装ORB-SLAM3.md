
# 预准备：安装 Ubuntu 20.04.5 LTS

因为*ROS1*支持到*Ubuntu 20.04.5 LTS*就停了

# 阶段一：系统更新与基础工具安装

首先，打开终端 (Ctrl+Alt+T)，进行系统更新和安装基础工具。

1.  更新系统源和软件：
    
    `sudo apt update && sudo apt upgrade -y`
    

2.  安装开发基础工具链 (包括C/C++编译器、Git、CMake等)：

    `sudo apt install -y build-essential cmake git pkg-config curl wget`
    

3.  安装Python3和Pip (Ubuntu 20.04默认已安装Python3，但需要确保pip):
    
    `sudo apt install -y python3-dev python3-pip python3-venv`
    

# 阶段二：安装编辑器

    VScode: https://code.visualstudio.com/Download
    Cursor: https://cursor.com/download

# 阶段三：安装ROS 1 Noetic

这是最关键的步骤之一，请仔细操作。

1.  设置ROS软件源：
    ```
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    ```

2.  设置ROS密钥：
    ```
    sudo apt install -y curl
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    ```
    

3.  安装ROS Noetic桌面完整版 (包含ROS, rqt, rviz, 机器人通用库, 2D/3D模拟器)：
    ```
    sudo apt update
    sudo apt install -y ros-noetic-desktop-full
    ```
    

4.  配置环境变量：
    每次启动新终端时自动sourceROS setup脚本，将以下命令添加到 ~/.bashrc 文件末尾。
    ```
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
    source ~/.bashrc
    ```
    

5.  安装ROS编译工具和依赖管理工具：
    ```
    sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
    ```

6.  初始化rosdep：
    ```
    sudo rosdep init
    rosdep update
    ```
    

7.  验证安装：
    打开一个新终端，输入 roscore。如果成功运行，说明ROS核心安装成功。按 Ctrl+C 停止。

# 阶段四：安装必要的依赖库

ORB-SLAM3和ROS需要一些特定的库。

1.  安装Pangolin (用于可视化)：
    ```
    # 安装依赖
    sudo apt install -y libglew-dev libwayland-dev libxkbcommon-dev wayland-protocols libegl1-mesa-dev
    # 克隆并编译Pangolin
    cd ~
    git clone https://github.com/stevenlovegrove/Pangolin.git
    cd Pangolin
    mkdir build && cd build
    cmake ..
    make -j$(nproc) # $(nproc)代表你CPU的核心数，编译更快
    sudo make install
    ```
    

2.  安装OpenCV (ROS Noetic自带OpenCV4，但ORB-SLAM3需要开发头文件)：
    ```
    sudo apt install -y libopencv-dev
    ```
    

3.  安装Eigen3 (线性代数库)：
    ```
    sudo apt install -y libeigen3-dev
    ```
    


# 阶段五：克隆和编译ORB-SLAM3

现在来安装你的项目核心——ORB-SLAM3。

```
​Monocular​：单目相机模式（仅一个摄像头）
​Monocular-Inertial​：单目+惯性测量单元（IMU）模式
​RGB-D​：RGB-D相机（如Kinect，提供深度信息）
​RGB-D-Inertial​：RGB-D+IMU模式
​Stereo​：双目相机模式
​Stereo-Inertial​：双目+IMU模式
```

1.  克隆ORB-SLAM3仓库：
    ```
    cd ~
    git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ORB_SLAM3
    ```
    

2.  编译第三方依赖：
    ```
    cd ORB_SLAM3
    chmod +x build.sh
    ./build.sh
    ```
    

3.  编译ORB-SLAM3本身：
    上一步的脚本也会编译ORB-SLAM3。如果一切顺利，你会在 ~/ORB_SLAM3/ 目录下看到生成的 libORB_SLAM3.so 库文件，以及在 Examples/ 目录下看到各种可执行文件，如 Monocular/mono_tum。

4.  （可选）构建ROS版本：
    如果你想在ROS节点中运行ORB-SLAM3，需要额外编译它的ROS包。
    ```
    chmod +x build_ros.sh
    ./build_ros.sh
    ```
    
    编译成功后，你需要将ORB_SLAM3的ROS包路径添加到ROS环境变量中。将下面这行添加到你的 ~/.bashrc 中（注意修改 $HOME 为你的实际家目录路径，如果不同的话）：
    ```
    echo "export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:$HOME/ORB_SLAM3/Examples/ROS" >> ~/.bashrc
    source ~/.bashrc
    ```
    

# 阶段六：测试ORB-SLAM3（单目）

我们使用TUM数据集的一个序列进行测试。

1.  下载数据集：
    ```
    cd ~
    wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
    tar -xzvf rgbd_dataset_freiburg1_xyz.tgz
    ```

2.  运行ORB-SLAM3单目示例：
    ```
    cd ORB_SLAM3
    ./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml ~/rgbd_dataset_freiburg1_xyz
    ```
    
    如果一切成功，Pangolin窗口将会打开，并开始运行SLAM，你可以看到地图点和相机轨迹的实时重建。



方向B：在Gazebo仿真环境中进行SLAM

1.  安装Gazebo：
    ROS Noetic桌面完整版通常已经包含了Gazebo。可以通过 gazebo --version 检查。

2.  安装TurtleBot3仿真包 (一个非常流行的机器人仿真模型)：
    ```
    sudo apt install -y ros-noetic-turtlebot3* gazebo11*
    ```
    

3.  设置TurtleBot3模型：
    ```
    echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
    source ~/.bashrc
    ```
    

4.  启动仿真世界：
    ```
    roslaunch turtlebot3_gazebo turtlebot3_world.launch
    ```
    
    这会启动Gazebo和一个带有摄像头的TurtleBot3机器人。

5.  运行ORB-SLAM3：
    同样，你需要先找到Gazebo摄像头发布的话题名（通常是 /camera/rgb/image_raw），然后修改ORB-SLAM3的启动文件或直接在执行命令中重映射话题。
    rosrun ORB_SLAM3 Mono Vocabulary/ORBvoc.txt ~/ORB_SLAM3/Examples/Monocular/TUM1.yaml _image_transport:=compressed /camera/image_raw:=/camera/rgb/image_raw
    
