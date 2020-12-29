#ifndef _BASE_TYPE_H_
#define _BASE_TYPE_H_

#include <memory>
#include <Eigen/Dense>

namespace MsfLocalization {
// IMU数据结构 
struct ImuData {
    double timestamp;      // In second.

    Eigen::Vector3d acc;   // Acceleration in m/s^2
    Eigen::Vector3d gyro;  // Angular velocity in radian/s.
    Eigen::Quaterniond rotate;  
};

using ImuDataPtr = std::shared_ptr<ImuData>;

// 轮速的数据结构
struct wheelSpeedData 
{
    double timestamp;      // In second.
    // 转速 弧度/时间 
    double left_speed;
    double right_speed;
};
using wheelSpeedDataPtr = std::shared_ptr<wheelSpeedData>;

// GNSS数据结构
struct GpsPositionData
{
    double timestamp;     // In second.
    int status;           // gps的状态  
    Eigen::Vector3d lla;  // Latitude in degree, longitude in degree, and altitude in meter.
    Eigen::Vector3d enu = {0., 0., 0.};  // 局部enu系坐标   初始化后通过转换获得 
    Eigen::Matrix3d cov;  // Covariance in m^2.     观测协方差 重要 !!!!!!!
};
using GpsPositionDataPtr = std::shared_ptr<GpsPositionData>;

// 估计的状态   
class State 
{
    public:
        double timestamp;
        // P,V,Q,bas, bgs
        // 下面P,R为局部坐标系下位姿
        Eigen::Vector3d P;         // The original point of the IMU frame in the Global frame.
        Eigen::Vector3d V;         // The velocity original point of the IMU frame in the Global frame.
        Eigen::Matrix3d R;         // The rotation from the IMU frame to the Global frame.
        Eigen::Vector3d acc_bias;  // The bias of the acceleration sensor.
        Eigen::Vector3d gyro_bias; // The bias of the gyroscope sensor.

        // Covariance.
        Eigen::Matrix<double, 15, 15> cov;
};

}  // ImuGpsLocalization

#endif   