#ifndef STATES_ESTIMATOR_HPP
#define STATES_ESTIMATOR_HPP

#include <memory>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pclomp/ndt_omp.h>
#include <pcl/filters/voxel_grid.h>

#include <System_kinematics.hpp>
#include <ukf.hpp>
#include <ekf.hpp>

/**
 * @brief scan matching-based pose estimator
 */
template<typename T>
class StatesEstimator {
public:
  using PointT = pcl::PointXYZI;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  /**
   * @brief constructor
   * @param registration        registration method   即NDT
   * @param stamp               timestamp
   * @param pos                 initial position      初始变换
   * @param quat                initial orientation   初始旋转
   * @param cool_time_duration  during "cool time", prediction is not performed
   */
  StatesEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const ros::Time& stamp, const Eigen::Vector3f& pos, const Eigen::Quaternionf& quat, double cool_time_duration = 1.0)
    : init_stamp(stamp),
      registration(registration),
      cool_time_duration(cool_time_duration)
  {
    // 状态噪声的协方差矩阵     16*16
    process_noise = Eigen::MatrixXf::Identity(16, 16);    // 单位阵
    process_noise.middleRows(0, 3) *= 1.0;                // 位置P的噪声方差
    process_noise.middleRows(3, 3) *= 1.0;                // 速度V的噪声方差
    process_noise.middleRows(6, 4) *= 0.5;                // Q的噪声方差      
    // 随机游走                
    process_noise.middleRows(10, 3) *= 1e-6;                              
    process_noise.middleRows(13, 3) *= 1e-6;
    // 控制的噪声    即IMU的测量噪声
    control_noise = Eigen::MatrixXf::Identity(6,6);
    control_noise.middleRows(0, 3) *= 0.0001;            // 加速度
    control_noise.middleRows(3, 3) *= 0.0001;            // 角速度 
    // 测量噪声协方差矩阵   Lidar的测量噪声      7*7
    Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);
    measurement_noise.middleRows(0, 3) *= 0.001;
    measurement_noise.middleRows(3, 4) *= 0.001;
    // 状态的均值初始化   设定的值是在地图坐标系的值
    Eigen::VectorXf mean(16);
    mean.middleRows(0, 3) = pos;        // 设定位置
    mean.middleRows(3, 3).setZero();    // 设定速度 初始为0 
    mean.middleRows(6, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());    // 6 - 9 设定姿态
    mean.middleRows(10, 3).setZero();     // 设定加速度偏置  初始化为0
    mean.middleRows(13, 3).setZero();     // 设定角速度偏置  初始化为0
    // 状态协方差矩阵初始化
    Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.3;

    PoseSystem system;    
    // 初始化UKF  
    ukf.reset(new UnscentedKalmanFilter<T, PoseSystem>(system, 16, 6, 7, process_noise, control_noise, measurement_noise, mean, cov));
    // 初始化ekf         初始位姿初始化       
    ekf.reset(new ExtendedKalmanFilter<T>(mean, cov));
  //  ekf.reset(new kkl::alg::ExtendedKalmanFilter<float>());
  }

  /**
   * @brief predict  根据IMU测得的加速度与角速度来预测
   * @param stamp    timestamp
   * @param acc      acceleration
   * @param gyro     angular velocity
   */
  void predict(const ros::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro) {
    if((stamp - init_stamp).toSec() < cool_time_duration || prev_stamp.is_zero() || prev_stamp == stamp) {
      prev_stamp = stamp;
      return;
    }
    // 计算前后时间间隔
    double dt = (stamp - prev_stamp).toSec();
    prev_stamp = stamp;

    ukf->setProcessNoiseCov(process_noise * dt);    // 返回过程噪声    process_noise的单位是s  
    ukf->system.dt = dt;
    //ekf->dt = dt;
    //ekf->setProcessNoiseCov(dt);

    Eigen::VectorXf control(6);
    control.head<3>() = acc;
    control.tail<3>() = gyro;
    
    mean = ukf->predict_simple(control);
    //mean = ukf->predict_ext(control);
    //ukf->predict_markvo(control);
    //mean = ekf->predict(control);
  }

  /**
   * @brief correct    校正过程  
   * @param cloud   input cloud
   * @return cloud aligned to the globalmap
   */
  pcl::PointCloud<PointT>::Ptr correct(const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
    // 设定预测值
    init_guess.block<3, 3>(0, 0) = quat().toRotationMatrix();
    init_guess.block<3, 1>(0, 3) = pos();

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    registration->setInputSource(cloud);
    registration->align(*aligned, init_guess);
    // 获取NDT匹配的结果
    Eigen::Matrix4f trans = registration->getFinalTransformation();
    // 提取平移
    Eigen::Vector3f p = trans.block<3, 1>(0, 3);
    // 提取旋转
    Eigen::Quaternionf q(trans.block<3, 3>(0, 0));
    // coeffs()即获取四元数的 m_coeffs 即 (x,y,z,w )   x.dot(y) = y^t*x    预测的旋转点乘激光观测的结果 
    if(quat().coeffs().dot(q.coeffs()) < 0.0f) {
      q.coeffs() *= -1.0f;
    }
    // 设定观测值
    Eigen::VectorXf observation(7);
    observation.middleRows(0, 3) = p;
    observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z());

    // 执行校正步骤
    //mean = ukf->correct_simple(observation);
    //ukf->correct_markvo(observation);    
    mean = ukf->correct(observation);      // hdl原装校正
    //mean = ekf->correct(observation);
    return aligned;
  }

  /* getters */
  // 获取ukf的预测位置
  Eigen::Vector3f pos() const {

    return Eigen::Vector3f(mean[0], mean[1], mean[2]);
  }
   
  Eigen::Vector3f vel() const {

    return Eigen::Vector3f(mean[3], mean[4], mean[5]);
  }
  // 获取UKF的预测姿态
  Eigen::Quaternionf quat() const {

    return Eigen::Quaternionf(mean[6], mean[7], mean[8], mean[9]).normalized();
  }

  Eigen::Matrix4f matrix() const {
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<3, 3>(0, 0) = quat().toRotationMatrix();
    m.block<3, 1>(0, 3) = pos();
    return m;
  }

private:
  ros::Time init_stamp;         // when the estimator was initialized
  ros::Time prev_stamp;         // when the estimator was updated last time
  double cool_time_duration;    //
  VectorXt mean;  
  Eigen::MatrixXf process_noise;
  Eigen::MatrixXf control_noise;       // 控制信号的噪声
  std::unique_ptr<UnscentedKalmanFilter<float, PoseSystem>> ukf; 
  std::unique_ptr<ExtendedKalmanFilter<float>> ekf; 

  pcl::Registration<PointT, PointT>::Ptr registration;
};



#endif 
















