/**
 * @brief 基于滤波器的INS定位 
 * @author wenhao li
 * @date 2020/9/26
 * @note 本代码实现基础的IMU&GPS融合定位 
 **/

#include <mutex>
#include <memory>
#include <iostream>
#include <thread>
#include <queue>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>

#include <nmea_msgs/Sentence.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Path.h>

#include <pcl/filters/voxel_grid.h>
#include <pclomp/ndt_omp.h>

#include "states_estimator.hpp"
#include "base_type.h"
#include "tic_toc.h"
#include "initializer.h"
#include "filters/filter.hpp"

using namespace std;
using namespace MsfLocalization; 

namespace INS {

// ROS与底层算法连接的类
class InsLocalization
{
  public:
    
    InsLocalization() = delete;  

    InsLocalization(std::shared_ptr<Filter> const& filter)
    {
      nh = ros::NodeHandle("~");       // 初始化私有句柄  

      imu_sub_ = nh.subscribe("/imu/data", 200, &InsLocalization::imuCallback, this);
      gps_sub_ = nh.subscribe("/GNSS_data", 200, &InsLocalization::gnssCallback, this);
      
      // 状态发布
      
      // estimate_state_pub_ = nh.advertise<nav_msgs::Odometry>("/odom", 5, false);
      pose_pub_ = nh.advertise<nav_msgs::Odometry>("/odom", 5);
      gps_path_pub_ = nh.advertise<nav_msgs::Path>("/gps_path", 5);
      fused_path_pub_ = nh.advertise<nav_msgs::Path>("/fused_path", 5);
      
      double acc_noise, gyro_noise, acc_bias_noise, gyro_bias_noise;    // 设置IMU噪声参数  默认值基本OK 
      nh.param("acc_noise",       acc_noise, 1e-2);   
      nh.param("gyro_noise",      gyro_noise, 1e-4);
      nh.param("acc_bias_noise",  acc_bias_noise, 1e-6);
      nh.param("gyro_bias_noise", gyro_bias_noise, 1e-8);
      // GPS, IMU 杆臂误差  
      double x, y, z;
      nh.param("I_p_Gps_x", x, 0.);
      nh.param("I_p_Gps_y", y, 0.);
      nh.param("I_p_Gps_z", z, 0.);
      const Eigen::Vector3d I_p_Gps(x, y, z);

      states_estimator_.reset(new StatesEstimator(acc_noise, gyro_noise, acc_bias_noise, gyro_bias_noise,
                              Eigen::Vector3d(0., 0., -9.81007), filter));  

    }

    void PublishStatePath(nav_msgs::Path &path, ros::Publisher &path_pub, State &state, string frame_id);

    void PublishPath(nav_msgs::Path &path, ros::Publisher &path_pub, Eigen::Vector3d const &P,
                                      Eigen::Quaterniond const &q, string frame_id); 

    void PublishOdometry(const ros::Time& stamp, const Eigen::Matrix4f& pose);

    void gnssCallback(const sensor_msgs::NavSatFixConstPtr& gps_msg_ptr);

    void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg);

    // 状态预测的线程 
    // 状态的预测  是通过 IMU和轮速传感器  完成的 
    void state_estimate();  

  private:
    ros::NodeHandle nh;                         // 初始化私有句柄  
    
    ros::Subscriber imu_sub_;   
    ros::Subscriber gps_sub_;      
  
    // 状态估计发布    PVQ  
    ros::Publisher pose_pub_;
    
    ros::Publisher gps_path_pub_;   
    ros::Publisher fused_path_pub_;

    nav_msgs::Path gps_path_;
    nav_msgs::Path fused_path_;

    // 互斥量  
    std::mutex imu_data_mutex_;
    std::mutex gnss_data_mutex_;

    std::queue<ImuData> imu_data_;                          // IMU队列 
    std::queue<GpsPositionData> gps_data_;  

    std::unique_ptr<StatesEstimator> states_estimator_;
    tf::TransformBroadcaster pose_broadcaster_;                // 这个玩意定义在函数内部  不能正常发布tf   直接做为全局变量又会和NodeHandle 冲突
    Initializer ins_init;  
    GeographicLib::LocalCartesian local_cartesian;

    bool use_imu_ = true;
    bool use_gnss_ = true;   
    bool invert_imu_ = false;
    bool initialize_ = false;

    double last_imu_t_ = 0.;  

};

void InsLocalization::PublishStatePath( nav_msgs::Path &path, ros::Publisher &path_pub, State &state, string frame_id) 
{
    path.header.frame_id = frame_id;
    path.header.stamp = ros::Time::now();  

    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    // 平移   
    pose.pose.position.x = state.P[0];
    pose.pose.position.y = state.P[1];
    pose.pose.position.z = state.P[2];
    // 旋转  
    const Eigen::Quaterniond G_q_I(state.R);
    pose.pose.orientation.x = G_q_I.x();
    pose.pose.orientation.y = G_q_I.y();
    pose.pose.orientation.z = G_q_I.z();
    pose.pose.orientation.w = G_q_I.w();

    path.poses.push_back(pose);

    path_pub.publish(path);  
}


void InsLocalization::PublishPath( nav_msgs::Path &path, ros::Publisher &path_pub, Eigen::Vector3d const& P, 
                                   Eigen::Quaterniond const& q, string frame_id) 
{
    path.header.frame_id = frame_id;
    path.header.stamp = ros::Time::now();  

    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    // 平移   
    pose.pose.position.x = P[0];
    pose.pose.position.y = P[1];
    pose.pose.position.z = P[2];
    // 旋转  
    pose.pose.orientation.x = q.x();
    pose.pose.orientation.y = q.y();
    pose.pose.orientation.z = q.z();
    pose.pose.orientation.w = q.w();

    path.poses.push_back(pose);

    path_pub.publish(path);  
}


/**
 * @brief publish odometry
 * @param stamp  timestamp
 * @param pose   odometry pose to be published
 */
void InsLocalization::PublishOdometry(const ros::Time& stamp, const Eigen::Matrix4f& pose) 
{
    // broadcast the transform over tf
    geometry_msgs::TransformStamped odom_trans = matrix2tf(stamp, pose, "map", "velodyne");
    pose_broadcaster_.sendTransform(odom_trans);

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";

    odom.pose.pose.position.x = pose(0, 3);
    odom.pose.pose.position.y = pose(1, 3);
    odom.pose.pose.position.z = pose(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;

    odom.child_frame_id = "velodyne";
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    pose_pub_.publish(odom);
}


/**
 * @brief IMU数据回调
 * @details IMU主要用于位姿插值, 主要作用是1. 去畸变  2. 进行位姿估计   
 *          每次接受到数据, 流程是: 1.如果初始化完成那么通过中值积分进行对位姿进行预测     2. 将IMU数据保存到数据队列中 
 * @param imu_msg   
 */ 
void InsLocalization::imuCallback(const sensor_msgs::ImuConstPtr& imu_msg) 
{
  // 检查时间戳 看是否乱序    
  if (imu_msg->header.stamp.toSec() <= last_imu_t_)
  {
      ROS_WARN("imu message in disorder!");
      return;
  }
                            
  ImuData imu; 
  // 获得加速度
  double dx = imu_msg->linear_acceleration.x;
  double dy = imu_msg->linear_acceleration.y;
  double dz = imu_msg->linear_acceleration.z;
  imu.acc = {dx, dy, dz};
  // 获得角速度 
  double rx = imu_msg->angular_velocity.x;
  double ry = imu_msg->angular_velocity.y;
  double rz = imu_msg->angular_velocity.z;
  imu.gyro = {rx, ry, rz};
  imu.timestamp = imu_msg->header.stamp.toSec();

  imu_data_mutex_.lock();
  imu_data_.push(imu);                   // 直接放入队列中
  imu_data_mutex_.unlock();   

  // 首先判断是否初始化    如果初始化  那么 通过IMU 进行预测  
  {
   // std::lock_guard<std::mutex> lg(pose_estimator_mutex_);
    // 如果初始化完成了   那么实时的对IMU数据进行预测 
    if (initialize_)
    {   
      double t = imu_msg->header.stamp.toSec();
      // 这里 latest_time在update()初始化 
      double dt = t - last_imu_t_;
      last_imu_t_ = t;
      
      // IMU航迹推算
      //states_estimator_->StatesForwardPropagation(linear_acceleration, angular_velocity, dt);  
      std_msgs::Header header = imu_msg->header;
      header.frame_id = "world";
      // pubLatestOdometry(header.stamp.toSec()); // 发布里程计信息，发布频率很高（与IMU数据同频），每次获取IMU数据都会及时进行更新，而且发布的是当前的里程计信息。
      // 还有一个pubOdometry()函数，似乎也是发布里程计信息， 有延迟，而且频率也不高（至多与激光同频）
    }
    else   // 如果没有初始化   那么要保证 imu_data 中的测量数量不超过100个 
    {
      if(imu_data_.size()>100)
      {
        imu_data_.pop();   // 最早的元素弹出   
      }
    }
  }
}   


/**
 * @brief GNSS回调函数  接收到GNSS数据后直接完成update
 * @details 初始化: 如果有点云地图  1. 点云地图和GNSS坐标对齐  直接用GPS赋初值   2. 没对齐, 首先获得点云地图的定位信息,与GPS的定位信息, 将两个坐标系对齐 
 *          使用策略 1. 当激光有效时, 不需要用GPS   2. 用GPS给定位初值    3. 当激光失效时  使用GNSS进行 校正 
 */
void InsLocalization::gnssCallback(const sensor_msgs::NavSatFixConstPtr& gps_msg_ptr)
{
  gnss_data_mutex_.lock();  
  // 检查 gps_状态 .
  if (gps_msg_ptr->status.status != 2) {
    // LOG(WARNING) << "[GpsCallBack]: Bad gps message!";
    return;
  }

  GpsPositionData gps_data;
  gps_data.timestamp = gps_msg_ptr->header.stamp.toSec();
  gps_data.status = gps_msg_ptr->status.status;
  gps_data.lla << gps_msg_ptr->latitude,     // 纬度
                  gps_msg_ptr->longitude,    // 经度
                  gps_msg_ptr->altitude;     // 高度
  gps_data.cov = Eigen::Map<const Eigen::Matrix3d>(gps_msg_ptr->position_covariance.data());     // 协方差   通过Map转换为 Matrix3d

  gnss_data_mutex_.unlock();
  
  if(!initialize_)
  {
    gps_data_.push(gps_data);
    return;  
  }
  
  // 求出ENU坐标 
  local_cartesian.Forward(gps_data.lla(0), gps_data.lla(1), gps_data.lla(2), 
                            gps_data.enu[0], gps_data.enu[1], gps_data.enu[2]);

  gps_data_.push(gps_data);

  // 发布gps轨迹
  PublishPath(gps_path_, gps_path_pub_, gps_data.enu, Eigen::Quaterniond{1, 0, 0, 0}, "origin"); 
}



// 状态预测的线程 
// 状态的预测  是通过 IMU和轮速传感器  完成的 
void InsLocalization::state_estimate()
{
    TicToc tt;
    tt.tic();  
    while (1)
    {
        // 下面的处理 要注意一定要是初始化完成之后
        if (initialize_)
        {
            // 如果数据容器容器中包含数据
            if (!gps_data_.empty() || !imu_data_.empty())
            {
                // LOG(INFO) << "imu nums: " << imu_data_.size() << "gps nums: " << gps_data_.size();
                // 找到容器中 时间戳最老的进行处理
                int idx = 0;
                double min_timestamp = numeric_limits<double>::max();
                if (!gps_data_.empty())
                {
                    min_timestamp = gps_data_.front().timestamp;
                    idx = 1;
                }
                if (!imu_data_.empty())
                {
                    if (imu_data_.front().timestamp < min_timestamp)
                    {
                        min_timestamp = imu_data_.front().timestamp;
                        idx = 2;
                    }
                }
                // LOG(INFO) << "process interval: " << tt.toc();
                // 当前处理的传感器
                switch (idx)
                {
                    case 1: // 最老的数据是gps
                    {
                        GpsPositionData gps_data = gps_data_.front();
                        gps_data_.pop();
                        // gps数据执行状态更新
                        states_estimator_->ProcessGPSData(gps_data);
                        break;
                    }
                    case 2: // 最老的数据是imu
                    {
                        // 如果初始化没有完成   不处理IMU的数据
                        ImuData imu_data = imu_data_.front();
                        imu_data_.pop();
                        // imu数据执行预测
                        states_estimator_->ProcessImuData(imu_data);

                        State &state = states_estimator_->GetCurrentState();
                        PublishStatePath(fused_path_, fused_path_pub_, state, "origin");
                        break;
                    }
                }
                // tt.tic();  
            }
        }
        else
        {
            // 有gps数据就初始化
            if (!gps_data_.empty())
            {
                // 加锁
                std::lock_guard<std::mutex> lg_gps(gnss_data_mutex_);
                std::lock_guard<std::mutex> lg_imu(imu_data_mutex_);
                GpsPositionData gps_data = gps_data_.front();
                gps_data_.pop();
                State &state = states_estimator_->GetCurrentState();
                if (ins_init.InsInitialize(gps_data, imu_data_, state, states_estimator_->last_imu_))
                {
                    states_estimator_->last_gps_time_ = gps_data.timestamp;
                    // 初始化GPS
                    local_cartesian.Reset(gps_data.lla(0), gps_data.lla(1), gps_data.lla(2));
                    initialize_ = true;
                    LOG(INFO) << "initialize OK " << std::endl
                              << "gps time stamp: " << std::setprecision(15) << states_estimator_->last_gps_time_
                              << "imu time stamp: " << states_estimator_->last_imu_.timestamp
                              << " P : " << state.P.transpose() << " V: " << state.V.transpose()
                              << " acc_bias: " << state.acc_bias.transpose() << " gyro_bias : " << state.gyro_bias.transpose()
                              << " R: " << std::endl
                              << state.R << std::endl
                              << "imu nums: " << imu_data_.size() << "gps nums: " << gps_data_.size();
                }
            }
        }
        // 延时
        std::chrono::milliseconds dura(1);
        std::this_thread::sleep_for(dura);
  }
}

} // INS


using namespace INS;   

int main(int argc, char **argv)
{
    FLAGS_log_dir = "/home/mini/code/localization_ws/src/msf_localization/LOG"; 
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;  //打印到日志同时是否打印到控制台 

    ros::init (argc, argv, "localization_node");   
    ROS_INFO("Started Lidar MsfLocalization node");

    std::shared_ptr<Filter> filter(new eskf());
    InsLocalization InsLocalization{filter};
    // ros::spin();                       // 单线程调用回调函数     
    ros::MultiThreadedSpinner spinner(2); // 多线程回调函数    Use 4 threads 
    std::thread process{&InsLocalization::state_estimate, &InsLocalization};      // 类成员函数 作为线程函数  
    spinner.spin();                       // spin() will not return until the node has been shutdown
    return 0;
}





