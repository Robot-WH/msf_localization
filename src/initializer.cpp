
#include <iostream>
#include <iomanip>
#include "initializer.h"

/**
 * @brief 多传感器融合定位 初始化类  
 * 
 * 
 */
using namespace std; 

namespace MsfLocalization {

Initializer::Initializer() {}


/**
 * @brief GPS数据来了后 进行初始化流程  
 * @param gps_data 最小的gps数据
 * @param imu_data imu数据的几何
 * @param[out] state 更新估计的状态
 * @param[out] last_imu 更新最后的imu数据  
 */  
bool Initializer::InsInitialize(GpsPositionData const& gps_data,  std::queue<ImuData> &imu_data, State &state, ImuData &last_imu) 
{
    // 首先保证足够的IMU数据   
    if (imu_data.size() < kImuDataBufferLength) 
    {
        LOG(WARNING) << "[AddGpsPositionData]: No enought imu data! imu num: "<<imu_data.size();
        return false;
    }
    // 检查最新的imu数据与GPS数据是否距离太远 
    // TODO: synchronize all sensors.
    if (std::abs(gps_data.timestamp - imu_data.back().timestamp) > 0.1) 
    {   
        // 清空imu
        while (!imu_data.empty()) 
            imu_data.pop();
        LOG(ERROR) << "[AddGpsPositionData]: Gps and imu timestamps are not synchronized!  abs(gps_data.timestamp - imu_data.back().timestamp) > 0.2";
        return false;
    }
    
    vector<ImuData> prepared_imus;
    int imu_num = imu_data.size();  
    // 准备用于初始化的数据   只将时间早于GPS时间的IMU数据进行初始化
    for (int i = 0; i < imu_num; i++)
    {   
        if(imu_data.front().timestamp>gps_data.timestamp)
            break;
        //std::cout << setprecision(15) << "gps_data.timestamp: " << gps_data.timestamp << " ,"
        //          << " imu time: " << imu_data.front().timestamp << std::endl;
        prepared_imus.push_back(imu_data.front());
        imu_data.pop();
    }
    
    if(prepared_imus.empty())
        return false;
    
    last_imu = prepared_imus.back(); 

    // 再检查用于初始化的最后一个IMU时间是否与GPS靠的足够近
    if (std::abs(gps_data.timestamp - last_imu.timestamp) > 0.1) 
    {
        LOG(ERROR) << "[AddGpsPositionData]: Gps and imu timestamps are not synchronized!  abs(gps_data.timestamp - last_imu.timestamp) > 0.5 ";
        return false;
    }

    // 初始状态的设定 
    // Set timestamp 
    state.timestamp = gps_data.timestamp;
    // Set initial mean.
    state.P.setZero();
    // We have no information to set initial velocity. 
    // So, just set it to zero and given big covariance.
    state.V.setZero();
   
    // 利用重力求解旋转
    if (!compute_R_FromImuData(state.R, prepared_imus)) 
    {
        LOG(WARNING) << "[AddGpsPositionData]: Failed to compute G_R_I!";
        return false;
    }
    // 陀螺仪偏置  用平均初始化 
    Eigen::Vector3d sum_gyro = {0., 0., 0.};  
    // 角速度偏置初始化
    for(ImuData& imu : prepared_imus)
    {
        sum_gyro += imu.gyro;
    }
    state.gyro_bias = sum_gyro / prepared_imus.size();
    // Set bias to zero.  加速度直接设置偏置为0   
    state.acc_bias.setZero();
    // 下面协方差是如何确定的???   通过数据手册   或者标定                 
    // Set covariance.
    state.cov.setZero();
    state.cov.block<3, 3>(0, 0) = 100. * Eigen::Matrix3d::Identity();      // position std: 10 m
    state.cov.block<3, 3>(3, 3) = 100. * Eigen::Matrix3d::Identity();      // velocity std: 10 m/s
    // roll pitch std 10 degree.   pitch, roll 由于 存在重力的观测 所以 比较准确噪声设置小一点  
    state.cov.block<2, 2>(6, 6) = 10. * kDegreeToRadian * 10. * kDegreeToRadian * Eigen::Matrix2d::Identity();    // 10度左右的误差 
    // 而yaw由于不可观  所以噪声设置大一点  
    state.cov(8, 8)             = 100. * kDegreeToRadian * 100. * kDegreeToRadian;          // yaw std: 100 degree.
    // 初始imu bias的协方差  
    // Acc bias.  加速度偏置的噪声大一点  0.1m的误差  
    state.cov.block<3, 3>(9, 9) = 0.01 * Eigen::Matrix3d::Identity();
    // Gyro bias.  陀螺仪的偏置噪声小一点      0.01弧度的误差 = 0.57度
    state.cov.block<3, 3>(12, 12) = 0.0001 * Eigen::Matrix3d::Identity();

    return true;
}

/**
 * @brief 初始化时  通过重力测量恢复旋转  
 * @param matrix_rot Rwi   即IMU的局部坐标系->世界坐标系UTM 的旋转  
 */ 
bool Initializer::compute_R_FromImuData(Eigen::Matrix3d &matrix_rot, vector<ImuData> const& imu_data) 
{
    
    // Compute mean and std of the imu buffer.   首先对加速度进行均值滤波  
    Eigen::Vector3d sum_acc(0., 0., 0.);
    for (const auto data : imu_data) 
    {
        sum_acc += data.acc;
    }
    const Eigen::Vector3d mean_acc = sum_acc / (double)imu_data.size();
    // 初始化需要静止   因此通过求方差判断运动是否过于剧烈  
    Eigen::Vector3d sum_err2(0., 0., 0.);
    for (const auto data : imu_data) 
    {
        sum_err2 += (data.acc - mean_acc).cwiseAbs2();    // 绝对值的平方    
    }
    // 求标准差 sigma    
    const Eigen::Vector3d std_acc = (sum_err2 / (double)imu_data.size()).cwiseSqrt();

    if (std_acc.maxCoeff() > kAccStdLimit) 
    {
        LOG(WARNING) << "[compute_R_FromImuData]: Too big acc std: " << std_acc.transpose();
        return false;
    }
    LOG(INFO) << "[compute_R_FromImuData]: acc is steady: " << std_acc.transpose();

    // Compute rotation.
    // Please refer to 
    // https://github.com/rpng/open_vins/blob/master/ov_core/src/init/InertialInitializer.cpp
    
    // Three axises of the ENU frame in the IMU frame.
    // z-axis.  这个就是(z轴)重力在IMU坐标系的投影值了
    const Eigen::Vector3d& z_axis = mean_acc.normalized(); 

    // 下面求世界坐标系的 x-axis. 在IMU坐标系的分量  
    Eigen::Vector3d x_axis = 
        Eigen::Vector3d::UnitX() - z_axis * z_axis.transpose() * Eigen::Vector3d::UnitX();
    x_axis.normalize();    // 归一化   
  
    // 下面求世界坐标系的 y-axis. 在IMU坐标系的分量 
    Eigen::Vector3d y_axis = z_axis.cross(x_axis);
    y_axis.normalize();
    // Riw 为 world -> imu   
    Eigen::Matrix3d Riw;
    Riw.block<3, 1>(0, 0) = x_axis;
    Riw.block<3, 1>(0, 1) = y_axis;
    Riw.block<3, 1>(0, 2) = z_axis; 
    // IMU -> global  
    matrix_rot = Riw.transpose();
    LOG(INFO) << "initialize ok! world to imu rotate: " <<std::endl << matrix_rot;
    return true;
}

}  // namespace MsfLocalization