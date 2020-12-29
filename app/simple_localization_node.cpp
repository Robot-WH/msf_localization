/**
 * @brief 基于滤波器的3D定位 
 * @details 融合多传感器 轮速计 + IMU + GNSS + 3D激光匹配   
 *          1. GNSS的作用: 1. 重定位可以先利用GNSS信息, 没有GNSS的话就用点云   2. 有激光的时候不用GNSS定位, 因为激光的精度足够高, 增加 \
 *             GNSS反而可能范围影响精度.   3. 无激光则采用 GNSS + IMU + 轮速 融合 
 *          3. 激光正常的情况下 定位采用 激光+IMU+轮速 方案
 *          4. 激光匹配是重点   1. 直接与地图匹配 , 那么就直接简单用NDT去匹配就行     2. 不直接与地图匹配.  这种更好! 可以降低环境变化的影响, 提升鲁棒性, 采用LIO+局部子图NDT校正
 *                            LIO拟采用 ESKF滤波器进行融合     3. 点云重定位的实现   相似性描述子 + super4PCS  
 *          5. 其他  1. 结合3D点云在2D栅格上定位     2. 动态部分加载匹配地图  
 * @author wenhao li
 * @date 2020/9/26
 * @note 本代码实现基础的激光与地图直接匹配的轮速+IMU&激光&GPS融合定位 
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

#include <pcl/filters/voxel_grid.h>
#include <pclomp/ndt_omp.h>

#include "states_estimator.hpp"

using namespace std;

typedef pcl::PointXYZI PointT;

namespace MsfLocalization {
// ROS与底层算法连接的类
class msfLocalization
{
  public:

    msfLocalization()
    {
      nh = ros::NodeHandle("~");       // 初始化私有句柄  
      initialize(nh);
      // 接收点云   接收队列设置为200   防止数据丢失        
      pointcloud_sub_ =  nh.subscribe<sensor_msgs::PointCloud2> ("/processed_points", 200, &msfLocalization::pointcloudCallback, this);       
        
      imu_sub_ = nh.subscribe("/imu/data", 200, &msfLocalization::imuCallback, this);
      gps_sub_ = nh.subscribe("/GNSS_data", 200, &msfLocalization::gnssCallback, this);
      
      SetMap_sub_ = nh.subscribe("/map", 2, &msfLocalization::mapCallback, this);                 // 地图接收
      
      // 状态发布
      
      // estimate_state_pub_ = nh.advertise<nav_msgs::Odometry>("/odom", 5, false);
      aligned_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 5);
      pose_pub_ = nh.advertise<nav_msgs::Odometry>("/odom", 5);

      // 外参
      Lidar2Imu_ = Eigen::Matrix4f::Identity();
      ENU2MAP_ = Eigen::Matrix3f::Identity();


    }

    geometry_msgs::TransformStamped matrix2tf(const ros::Time& stamp, const Eigen::Matrix4f& pose,
                                                      const std::string& frame_id, const std::string& child_frame_id);
                                                
    void publishOdometry(const ros::Time& stamp, const Eigen::Matrix4f& pose);

    void pubLatestOdometry(double const &t);   

    void pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& pointcloud);

    void gnssCallback(const sensor_msgs::NavSatFixConstPtr& gps_msg_ptr);

    void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg);

    void mapCallback(const sensor_msgs::PointCloud2ConstPtr& points_msg);

    void initialize(ros::NodeHandle& nh);

    // 状态预测的线程 
    // 状态的预测  是通过 IMU和轮速传感器  完成的 
    void state_estimate();  

    // std::queue<sensor_msgs::ImuConstPtr> & GetImuQue()
    // {
    //   return imu_data_;  
    // }

    // std::queue<sensor_msgs::ImuConstPtr> & GetWheelSpeedQue()
    // {
    //   return wheelSpeed_data_;  
    // }

  private:
    ros::NodeHandle nh;                         // 初始化私有句柄  
    
    ros::Subscriber pointcloud_sub_;
    ros::Subscriber imu_sub_;   
    ros::Subscriber gps_sub_;            
    ros::Subscriber SetMap_sub_;           // 匹配地图 
  
    // 状态估计发布    PVQ  
    ros::Publisher estimate_state_pub_;  
    ros::Publisher aligned_pub_;
    ros::Publisher pose_pub_;
    ros::Publisher pub_latest_odometry_;
    
    // 互斥量  
    std::mutex imu_data_mutex_;
    std::mutex gnss_data_mutex_;
    std::mutex lidar_data_mutex_;
    std::mutex state_estimator_mutex_;

    std::queue<ImuData> imu_data_;                          // IMU队列 
    std::queue<wheelSpeedData> wheelSpeed_data_;            // 轮速队列    轮速度消息  
    std::queue<GpsPositionData> gps_data_;  
    std::queue<pcl::PointCloud<PointT>> lidar_data_;  

    pcl::Registration<PointT, PointT>::Ptr registration_;      // 匹配方法  
    pcl::PointCloud<PointT>::Ptr Matched_map_;
    std::unique_ptr<StatesEstimator> states_estimator_;
    tf::TransformBroadcaster pose_broadcaster_;                // 这个玩意定义在函数内部  不能正常发布tf   直接做为全局变量又会和NodeHandle 冲突   

    Eigen::Matrix4f Lidar2Imu_;       // LIDAR - IMU 外参 
    Eigen::Matrix3f ENU2MAP_;         // ENU - MAP外参     IMU测量的都是ENU系下的数据  需要转换到Map系下  
    
    bool use_lidar_ = true;           
    bool use_imu_ = true;
    bool use_gnss_ = true;   
    bool use_wheelSpeed_ = false;     // 默认不使用 轮速 
    bool invert_imu_ = false;
    bool initialize_ = false;

    uint8_t lidar_status_ = 0;   // 0: 无激光   1: 效果不佳   2: 正常工作  
    double last_imu_t_ = 0.;  
};

/**
 * @brief convert a Eigen::Matrix to TransformedStamped
 * @param stamp           timestamp
 * @param pose            pose matrix
 * @param frame_id        frame_id
 * @param child_frame_id  child_frame_id
 * @return transform
 */
geometry_msgs::TransformStamped msfLocalization::matrix2tf(const ros::Time& stamp, const Eigen::Matrix4f& pose, 
                                                           const std::string& frame_id, const std::string& child_frame_id) 
{
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
void msfLocalization::publishOdometry(const ros::Time& stamp, const Eigen::Matrix4f& pose) 
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
 * @brief 点云的回调函数 
 * @param stamp  timestamp
 * @param pose   odometry pose to be published   
 */
void msfLocalization::pointcloudCallback(const sensor_msgs::PointCloud2ConstPtr& pointcloud)
{   
    /*
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
    if(!use_imu_) {
      // 如果没有IMU 
      states_estimator->predict(stamp, Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero());
    } else {
      // 有IMU   
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.front();
      // 遍历两帧激光间积累的所有IMU数据   
      for(imu_iter; imu_iter != imu_data.back(); imu_iter++) {
        if(stamp < (*imu_iter)->header.stamp) {
          break;
        }
        // 读取加速度 和 角速度
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        // 方向符号
        double gyro_sign = invert_imu_ ? -1.0 : 1.0;
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
    */
}


/**
 * @brief IMU数据回调
 * @details IMU主要用于位姿插值, 主要作用是1. 去畸变  2. 进行位姿估计   
 *          每次接受到数据, 流程是: 1.如果初始化完成那么通过中值积分进行对位姿进行预测     2. 将IMU数据保存到数据队列中 
 * @param imu_msg   
 */ 
void msfLocalization::imuCallback(const sensor_msgs::ImuConstPtr& imu_msg) 
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
void msfLocalization::gnssCallback(const sensor_msgs::NavSatFixConstPtr& gps_msg_ptr)
{

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

  gps_data_.push(gps_data);
  
  /*
  if(!initialize_)          // 如果没有初始化   进行初始化
  {
    if(!states_estimator_->InitializeByImuAndGps())     
      initialize_ = true;  
  }

  // 进行状态的updata 
  
  // 检查雷达的状态 
  if(lidar_status_ == 2)    // 雷达正常则不使用GNSS 
    return;   
  */           
}


// 匹配用的全局地图设置回调
void msfLocalization::mapCallback(const sensor_msgs::PointCloud2ConstPtr& points_msg) 
{
  ROS_INFO("map received!");
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
  pcl::fromROSMsg(*points_msg, *cloud);
  Matched_map_ = cloud;
  registration_->setInputTarget(Matched_map_);        // 设置匹配地图   
}


void msfLocalization::initialize(ros::NodeHandle& nh)
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
    registration_ = ndt;

    // 传感器配置  
    use_imu_ = nh.param<bool>("use_imu", true);            // 使用IMU
    invert_imu_ = nh.param<bool>("invert_imu", false);     // true 
    cout<<"IMU ENABLE: "<<use_imu_<<" invert: "<<invert_imu_<<endl;
    
    // 位姿初始化  
    // launch文件的设定是   直接通过下面参数给初值
    if(nh.param<bool>("specify_init_pose", true)) {
      ROS_INFO("initialize pose estimator with specified parameters!!");
      /*
      states_estimator_.reset(new StatesEstimator<float>(registration_,
        ros::Time::now(),
        Eigen::Vector3f(nh.param<double>("init_pos_x", 0.0), nh.param<double>("init_pos_y", 0.0), nh.param<double>("init_pos_z", 0.0)),
        Eigen::Quaternionf(nh.param<double>("init_ori_w", 1.0), nh.param<double>("init_ori_x", 0.0), nh.param<double>("init_ori_y", 0.0), nh.param<double>("init_ori_z", 0.0)),
        nh.param<double>("cool_time_duration", 0.5)
      ));
      */
    }  
}

// 状态预测的线程 
// 状态的预测  是通过 IMU和轮速传感器  完成的 
void msfLocalization::state_estimate()
{
  while(1)
  { 
    // 下面的处理 要注意一定要是初始化完成之后
    if(initialize_)
    {
      // 如果数据容器容器中包含数据
      if(!gps_data_.empty()||!imu_data_.empty()||!wheelSpeed_data_.empty()||!lidar_data_.empty())
      {
        // 找到容器中 时间戳最老的进行处理   
        int idx = 0;
        double min_timestamp = 0.;  
        if (!gps_data_.empty())
        {
          min_timestamp = gps_data_.front().timestamp;
          idx = 1;  
        }
        if (!imu_data_.empty())
        {
          if(imu_data_.front().timestamp<min_timestamp)
          {
            min_timestamp = imu_data_.front().timestamp;
            idx = 2;  
          }
        }
        if (!wheelSpeed_data_.empty())
        {
          if(wheelSpeed_data_.front().timestamp<min_timestamp)
          {
            min_timestamp = wheelSpeed_data_.front().timestamp;
            idx = 3;  
          }
        }
        if (!lidar_data_.empty())
        {
          if(lidar_data_.front().header.stamp<min_timestamp)
          {
            min_timestamp = lidar_data_.front().header.stamp;    // unit: ms
            idx = 4;  
          }
        }
        // 当前处理的传感器
        switch(idx)
        {
          case 1:    // 最老的数据是gps
          {
            GpsPositionData gps_data = gps_data_.front();
            gps_data_.pop();  
            // gps数据执行状态更新  
            break;
          }
          case 2:    // 最老的数据是imu
          {
            // 如果初始化没有完成   不处理IMU的数据
            ImuData imu_data = imu_data_.front();
            imu_data_.pop();  
            // imu数据执行预测 
          
            break;
          }
          case 3:    // 最老的数据是wheelspeed
          {
            break;
          }
          case 4:    // 最老的数据是lidar 
          {
            break;  
          }
        }

        // 首先保证初始化完成 才进行状态的预测
        if(initialize_)
        { // 轮速和IMU中需要有数据 
          if(!imu_data_.empty()||!wheelSpeed_data_.empty())   
          { 
            // 取出轮速与IMU最前的数据
            ImuData imu_ptr;
            wheelSpeedData wheelSpeed_ptr;
            // 如果只有轮速的信息
            if(imu_data_.empty())
            {
              wheelSpeed_ptr = wheelSpeed_data_.front();
              wheelSpeed_data_.pop();

            } // 如果只有IMU的数据  设置击中更新模式 1. 6Dof运动学   2. 
            else if(wheelSpeed_data_.empty())
            {
              imu_ptr = imu_data_.front();
              imu_data_.pop();
              // 用imu数据进行状态估计的预测环节  

            }
            else // 两个数据都有  即同时用轮速记与IMU进行预测  这时 IMU只用角速度 更新旋转 ,线速度认为不变, 轮速记更新线速度 ,角速度认为不变
            {
              wheelSpeed_ptr = wheelSpeed_data_.front();
              imu_ptr = imu_data_.front();
              // 比较两者时间戳  按先后顺序执行预测   

            }
          }
        }
      }
    }
    else    // 进行初始化,初始化和轮速无关 
    {       //            1. 有雷达的话, 初始化需要完成 (1). 确定激光初始在地图上的位姿, 无GPS情况下(地图无GPS信息)只能重定位算法求解(描述子+粗匹配+细匹配)
            //               如果有GPS数据, 而且建立的地图包含GPS的信息 那么可以用GPS数据以及IMU数据给定初值
            //            2. 匹配初始化完成后,   无GPS有IMU 求地图关于IMU的旋转 , 有GPS有IMU, 则GPS需要对其地图 
      if(!gps_data_.empty())     
      {
          
      }
      else
      {

      }
    }
  }
}

} // msfLocalization


using namespace MsfLocalization;   

int main(int argc, char **argv)
{
    ros::init (argc, argv, "localization_node");   
    ROS_INFO("Started Lidar MsfLocalization node");   
    msfLocalization msf_localization{}; 
    // ros::spin();                       // 单线程调用回调函数     
    ros::MultiThreadedSpinner spinner(2); // 多线程回调函数    Use 4 threads 
    std::thread process{&msfLocalization::state_estimate, &msf_localization};      // 类成员函数 作为线程函数  
    spinner.spin();                       // spin() will not return until the node has been shutdown
    return 0;
}





