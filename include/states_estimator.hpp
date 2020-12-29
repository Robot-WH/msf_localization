#ifndef STATES_ESTIMATOR_HPP
#define STATES_ESTIMATOR_HPP

#include <iostream>
#include <memory>
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pclomp/ndt_omp.h>
#include <pcl/filters/voxel_grid.h>

#include <glog/logging.h>

#include <System_kinematics.hpp>

#include "filters/filter.hpp"
#include "filters/eskf.hpp"
#include "filters/ukf.hpp"
#include "base_type.h"


namespace MsfLocalization {

using namespace std; 

/**
 * @brief 滤波器状态估计类 
 */
class StatesEstimator 
{
public:

  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
  
  StatesEstimator(double const& acc_noise, double const& gyro_noise, double const& acc_bias_noise, 
                  double const& gyro_bias_noise, Eigen::Vector3d const gravity, 
                  std::shared_ptr<Filter> const& filter) 
  : filter_(filter), acc_noise_(acc_noise), gyro_noise_(gyro_noise), acc_bias_noise_(acc_bias_noise), 
    gyro_bias_noise_(gyro_bias_noise),  gravity_(gravity)
  {
    // 状态初始化
    state_.timestamp = 0.0;
    state_.P = {0.0, 0.0, 0.0};
    state_.V = {0.0, 0.0, 0.0};
    state_.R = Eigen::Matrix3d::Zero();
    state_.acc_bias = {0.0, 0.0, 0.0};
    state_.gyro_bias = {0.0, 0.0, 0.0};
    state_.cov = MatrixXd::Zero(15,15);
  }

  /**
   * @brief constructor
   * @param registration        registration method   即NDT
   * @param stamp               timestamp
   * @param pos                 initial position      初始变换
   * @param quat                initial orientation   初始旋转
   * @param cool_time_duration  during "cool time", prediction is not performed
   */
  /*
  StatesEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const ros::Time& stamp, const Eigen::Vector3f& pos, const Eigen::Quaternionf& quat, double cool_time_duration = 1.0)
    : init_stamp(stamp),
      registration_(registration),
      cool_time_duration_(cool_time_duration)
  {
    /*
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
  */
  
  /**
   * @brief 处理IMU数据  
   * @details 用IMU数据进行预测 
   * @param stamp    timestamp
   * @param acc      acceleration
   * @param gyro     angular velocity
  */
  bool ProcessImuData(ImuData const& cur_imu)
  {
    LOG(INFO) << "imu predict !!  curr time: " << std::setprecision(15) << cur_imu.timestamp; 
    // 如果上一个处理的imu的时间 < 上一个处理的GPS数据  说明上一个处理的数据是GPS  需要插值出GPS时间处IMU的数据
    if(last_gps_time_>last_imu_.timestamp)
    {
      if(cur_imu.timestamp <= last_gps_time_)
      {
        last_imu_ = cur_imu;
        return false;  
      }
      //插值
      // 时间戳的对应关系如下图所示：
      //                                            current_time         t
      // *               *               *               *               *     （IMU数据）
      //                                                          |            （ 数据）
      //
      LOG(INFO) << "pre imu data time: " << std::setprecision(15) << last_imu_.timestamp;  
      LOG(INFO) << "last_gps_time_: " << std::setprecision(15) << last_gps_time_;  

      double dt_1 = last_gps_time_ - last_imu_.timestamp;    
      double dt_2 = cur_imu.timestamp - last_gps_time_;

      LOG(INFO) << "curr imu acc: " << cur_imu.acc.transpose() << " curr imu gyro: " << cur_imu.gyro.transpose() << std::endl
                << "last imu acc: " << last_imu_.acc.transpose() << " last imu gyro: " << last_imu_.gyro.transpose() << std::endl; 
      // 计算插值系数       这么来算    dx + (cx - dx)*dt_1/(dt_1+dt_2)
      double w1 = dt_2 / (dt_1 + dt_2);
      double w2 = dt_1 / (dt_1 + dt_2);
      // 插值出img处imu的近似值
      last_imu_.acc = w1 * last_imu_.acc + w2 * cur_imu.acc;
      last_imu_.gyro = w1 * last_imu_.gyro + w2 * cur_imu.gyro;
      last_imu_.timestamp = last_gps_time_;
      
      LOG(INFO) << "interpolation imu acc: " << last_imu_.acc.transpose() << " gyro: " << last_imu_.gyro.transpose();  
    }
    // 调用滤波器的IMU预测环节 
    filter_->PredictByImu(last_imu_, cur_imu, state_, gravity_, acc_noise_, gyro_noise_, acc_bias_noise_, gyro_bias_noise_);
    last_imu_ = cur_imu;  
    //LOG(INFO) << "IMU predict ! P: " << state_.P.transpose();
  }

  /**
   * @brief 处理GPS数据  
   * @details 用GPS数据进行校正 
   * @param stamp    timestamp
  */
  bool ProcessGPSData(GpsPositionData const& gps_data)
  {
    LOG(INFO) << " GPS correct!!  timestamp: "<< std::setprecision(15) << gps_data.timestamp;  
    filter_->UpdateByGps(gps_data, state_);
    last_gps_time_ = gps_data.timestamp;   // 记录当前GPS数据时间
    //LOG(INFO) << "GPS correct ! P: " << state_.P.transpose();
  }

  /**
   * @brief 处理激光数据  
   * @details 用激光数据进行校正 
   * @param stamp    timestamp
  */
  bool ProcessLidarData()
  {
  }


    // 通过IMU去进行  前向传播   
  void StatesForwardPropagation( Eigen::Vector3d const& angular_velocity, Eigen::Vector3d const& linear_acceleration, double const& dt)
  { 
    // 调用IMU运动模型计算 
    // imu_integration_->predict(angular_velocity, linear_acceleration, tmp_P_, tmp_V_, tmp_Q_, tmp_Ba_, tmp_Bg_, dt);
  }


  // 获取状态  可写
  State& GetCurrentState()
  {
    return state_; 
  }


protected:

public:
  double last_gps_time_; 
  ImuData last_imu_;
    
private:
  ros::Time init_stamp;         // when the estimator was initialized
  ros::Time prev_stamp;         // when the estimator was updated last time
  double cool_time_duration_;   // 冷却时间 

  std::shared_ptr<Filter> filter_;
  // 维护当前的状态 
  State state_;
  // 参数
  // IMU相关
  double const acc_noise_, gyro_noise_;                    // 测量噪声 
  double const acc_bias_noise_, gyro_bias_noise_;          // 随机游走噪声 
  Eigen::Vector3d const gravity_;                          // 重力 
};

} // namespace MsfLocalization 

#endif 
















