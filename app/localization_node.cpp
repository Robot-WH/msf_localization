#include <mutex>
#include <memory>
#include <iostream>
#include <thread>
 
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <pcl/filters/voxel_grid.h>
#include <pclomp/ndt_omp.h>

#include "states_estimator.hpp"




using namespace std;

typedef pcl::PointXYZI PointT;

class localization
{
    public:
    localization() 
    {
      nh = ros::NodeHandle("~");       // 初始化私有句柄  
      initialize(nh);
        // 接收点云   接收队列设置为200   防止数据丢失        
      pointcloud_sub =  nh.subscribe<sensor_msgs::PointCloud2> ("/processed_points", 200, &localization::pointcloud_callback, this);       
        
      if(use_imu) 
          imu_sub = nh.subscribe("/gpsimu_driver/imu_data", 200, &localization::imu_callback, this);
      SetMap_sub = nh.subscribe("/map", 2, &localization::map_callback, this);                 // 地图接收
      
      // 状态发布
      // estimate_state_pub = nh.advertise<nav_msgs::Odometry>("/odom", 5, false);
      aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 5);
      pose_pub = nh.advertise<nav_msgs::Odometry>("/odom", 5);

      // 外参
      Lidar2Imu = Eigen::Matrix4f::Identity();
      ENU2MAP = Eigen::Matrix3f::Identity();

    }
    

    geometry_msgs::TransformStamped matrix2transform(const ros::Time& stamp, const Eigen::Matrix4f& pose,
                                                     const std::string& frame_id, const std::string& child_frame_id);
                                                
    void publish_odometry(const ros::Time& stamp, const Eigen::Matrix4f& pose);         

    void pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& pointcloud);


    void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg);

    void map_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg);

    void initialize(ros::NodeHandle& nh);




    private:
       ros::NodeHandle nh;                         // 初始化私有句柄  
       
       ros::Subscriber pointcloud_sub;
       ros::Subscriber imu_sub;            
       ros::Subscriber SetMap_sub;           // 匹配地图 

       // 状态估计发布    PVQ  
       ros::Publisher estimate_state_pub;  
       ros::Publisher aligned_pub;
       ros::Publisher pose_pub;

       std::mutex imu_data_mutex ;
       std::mutex pose_estimator_mutex;

       std::vector<sensor_msgs::ImuConstPtr> imu_data;
       pcl::Registration<PointT, PointT>::Ptr registration;      // 匹配方法  
       pcl::PointCloud<PointT>::Ptr Matched_map;
       std::unique_ptr<StatesEstimator<float>> states_estimator;
       tf::TransformBroadcaster pose_broadcaster;                // 这个玩意定义在函数内部  不能正常发布tf   直接做为全局变量又会和NodeHandle 冲突   

       Eigen::Matrix4f Lidar2Imu;       // LIDAR - IMU 外参 
       Eigen::Matrix3f ENU2MAP;         // ENU - MAP外参     IMU测量的都是ENU系下的数据  需要转换到Map系下  

       bool use_imu = 0;
       bool invert_imu = 0;

};


/**
 * @brief convert a Eigen::Matrix to TransformedStamped
 * @param stamp           timestamp
 * @param pose            pose matrix
 * @param frame_id        frame_id
 * @param child_frame_id  child_frame_id
 * @return transform
 */
geometry_msgs::TransformStamped localization::matrix2transform(const ros::Time& stamp, const Eigen::Matrix4f& pose, const std::string& frame_id, const std::string& child_frame_id) {
    // 旋转矩阵转四元数   
    Eigen::Quaternionf quat(pose.block<3, 3>(0, 0));
    quat.normalize();
    geometry_msgs::Quaternion odom_quat;
    odom_quat.w = quat.w();
    odom_quat.x = quat.x();
    odom_quat.y = quat.y();
    odom_quat.z = quat.z();
    
    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = stamp;
    odom_trans.header.frame_id = frame_id;
    odom_trans.child_frame_id = child_frame_id;
    // 平移部分 
    odom_trans.transform.translation.x = pose(0, 3);
    odom_trans.transform.translation.y = pose(1, 3);
    odom_trans.transform.translation.z = pose(2, 3);
    odom_trans.transform.rotation = odom_quat;

    return odom_trans;
}

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  
void localization::publish_odometry(const ros::Time& stamp, const Eigen::Matrix4f& pose) {
    // broadcast the transform over tf
    geometry_msgs::TransformStamped odom_trans = matrix2transform(stamp, pose, "map", "velodyne");
    pose_broadcaster.sendTransform(odom_trans);

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

    pose_pub.publish(odom);
}
  
  

// 点云回调
void localization::pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& pointcloud)
{   
    
    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    // 等待初始化完成   
    // 初始化位姿  初始化全局地图
    if(!states_estimator) {
      ROS_ERROR("waiting for initial pose input!!");
      return;
    }
    if(!Matched_map) {
      ROS_ERROR("map has not been received!!");
      return;
    }
    // 获得时间戳
    const auto& stamp = pointcloud->header.stamp;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*pointcloud, *cloud);

    if(cloud->empty()) {
      ROS_ERROR("cloud is empty!!");
      return;
    }

    // 对于每一帧点云  定位时都执行 预测 -> 校正 
    // predict
    if(!use_imu) {
      // 如果没有IMU 
      states_estimator->predict(stamp, Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero());
    } else {
      // 有IMU   
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.begin();
      // 遍历两帧激光间积累的所有IMU数据   
      for(imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        if(stamp < (*imu_iter)->header.stamp) {
          break;
        }
        // 读取加速度 和 角速度
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        // 方向符号
        double gyro_sign = invert_imu ? -1.0 : 1.0;
        // 对于每一个IMU数据执行   预测    
        states_estimator->predict((*imu_iter)->header.stamp, Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      // 清空IMU队列  
      imu_data.erase(imu_data.begin(), imu_iter);
    }

     // correct
    auto t1 = ros::WallTime::now();
    auto aligned = states_estimator->correct(cloud);
    auto t2 = ros::WallTime::now();

//    processing_time.push_back((t2 - t1).toSec());
//    double avg_processing_time = std::accumulate(processing_time.begin(), processing_time.end(), 0.0) / processing_time.size();
    cout<<"matching time: " << (t2 - t1) * 1000.0 << " [msec]"<<endl;
    if(aligned_pub.getNumSubscribers()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
 //     aligned_pub.publish(aligned);
    }
    // 发布位姿
    publish_odometry(pointcloud->header.stamp, states_estimator->matrix());    
}

// IMU数据回调  
void localization::imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
  std::lock_guard<std::mutex> lock(imu_data_mutex);
  imu_data.push_back(imu_msg);                   // 直接放入队列中
}   

/*
// 点云回调
void localization::pointcloud_callback(const sensor_msgs::PointCloud2ConstPtr& pointcloud)
{   
    
    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    // 等待初始化完成   
    // 初始化位姿  初始化全局地图
    if(!states_estimator) {
      ROS_ERROR("waiting for initial pose input!!");
      return;
    }
    if(!Matched_map) {
      ROS_ERROR("map has not been received!!");
      return;
    }
    // 获得时间戳
    const auto& stamp = pointcloud->header.stamp;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*pointcloud, *cloud);

    if(cloud->empty()) {
      ROS_ERROR("cloud is empty!!");
      return;
    }

     // correct
    auto t1 = ros::WallTime::now();
    auto aligned = states_estimator->correct(cloud);
    auto t2 = ros::WallTime::now();

//    processing_time.push_back((t2 - t1).toSec());
//    double avg_processing_time = std::accumulate(processing_time.begin(), processing_time.end(), 0.0) / processing_time.size();
    cout<<"matching time: " << (t2 - t1) * 1000.0 << " [msec]"<<endl;
    if(aligned_pub.getNumSubscribers()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
 //     aligned_pub.publish(aligned);
    }
    // 发布位姿
    publish_odometry(pointcloud->header.stamp, states_estimator->matrix());    
}

// IMU数据回调       使用IMU进行预测   
void localization::imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
  std::lock_guard<std::mutex> lock(imu_data_mutex);
  imu_data.push_back(imu_msg);                            // 直接放入队列中

}

ros::Time predict_pretime = ros::Time(0);
#define predict_duration 0.02             // 50HZ   

void state_predict()
{
  while(1)
  {
    if(predict_pretime.toSec() == 0)
      predict_pretime = ros::Time::now();
    else
    {
      double diff_time = (ros::Time::now() - predict_pretime).toSec();     // 计算时间差
      if(diff_time>= predict_duration)
      { 
        if(!imu_data.empty()){
          while(!imu_data.empty())
          {
              // 对于每一帧点云  定位时都执行 预测 -> 校正 
              // predict
              if(use_imu) {
                // 有IMU   
                auto imu = imu_data.front();
                imu_data.pop();
                
                  if(stamp < (*imu_iter)->header.stamp) 
                    break;
                
                  // 读取加速度 和 角速度
                  const auto& acc = (*imu_iter)->linear_acceleration;
                  const auto& gyro = (*imu_iter)->angular_velocity;
                  // 方向符号
                  double gyro_sign = invert_imu ? -1.0 : 1.0;
                  // 对于每一个IMU数据执行   预测    
                  states_estimator->predict((*imu_iter)->header.stamp, Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
                }
              }
          }
        }
        else
        {
           states_estimator->predict(stamp, Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero());
        }
        
      }
    }
  }
}   */

// 匹配用的全局地图设置回调
void localization::map_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
  ROS_INFO("map received!");
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
  pcl::fromROSMsg(*points_msg, *cloud);
  Matched_map = cloud;
  registration->setInputTarget(Matched_map);        // 设置匹配地图   
}

void localization::initialize(ros::NodeHandle& nh)
{
    // NDT参数设置 
    double ndt_resolution = nh.param<double>("ndt_resolution", 1.0);     // 网格分辨率
    std::string ndt_neighbor_search_method = nh.param<std::string>("ndt_neighbor_search_method", "DIRECT7");  // 搜索方法？？？？？？？？？？？？？？
    pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
    ndt->setTransformationEpsilon(0.01);          // 最小迭代
    ndt->setResolution(ndt_resolution);           // 网格分辨率  
    if(ndt_neighbor_search_method == "DIRECT1") {
        ROS_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
    } else if(ndt_neighbor_search_method == "DIRECT7") {
        ROS_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
    } else {
        if(ndt_neighbor_search_method == "KDTREE") {
        ROS_INFO("search_method KDTREE is selected");
        } else {
        ROS_WARN("invalid search method was given");
        ROS_WARN("default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
    }
    registration = ndt;

    // 传感器配置  
    use_imu = nh.param<bool>("use_imu", true);            // 使用IMU
    invert_imu = nh.param<bool>("invert_imu", false);     // true 
    cout<<"IMU ENABLE: "<<use_imu<<" invert: "<<invert_imu<<endl;
    
    // 位姿初始化  
    // launch文件的设定是   直接通过下面参数给初值
    if(nh.param<bool>("specify_init_pose", true)) {
      ROS_INFO("initialize pose estimator with specified parameters!!");
      states_estimator.reset(new StatesEstimator<float>(registration,
        ros::Time::now(),
        Eigen::Vector3f(nh.param<double>("init_pos_x", 0.0), nh.param<double>("init_pos_y", 0.0), nh.param<double>("init_pos_z", 0.0)),
        Eigen::Quaternionf(nh.param<double>("init_ori_w", 1.0), nh.param<double>("init_ori_x", 0.0), nh.param<double>("init_ori_y", 0.0), nh.param<double>("init_ori_z", 0.0)),
        nh.param<double>("cool_time_duration", 0.5)
      ));
    }  
}


int main(int argc, char **argv)
{
    ros::init (argc, argv, "localization_node");   
    ROS_INFO("Started Lidar localization node");   
    localization loc; 
    //    ros::spin();               // 单线程调用回调函数     
    ros::MultiThreadedSpinner spinner(2); // Use 4 threads
    //std::thread process{state_predict}; 
    spinner.spin();                       // spin() will not return until the node has been shutdown
    return 0;
}





