
/**
 * @brief 滤波器基类
 * @details 后续可以派生出多种滤波器  如 eskf, ieskf ,ukf
 */

#ifndef FILTER_HPP
#define FILTER_HPP

#include "base_type.h"
#include "utility.hpp"

namespace MsfLocalization{

    class Filter
    {
        public:
        Filter()
        {}

        virtual ~Filter()
        {}
        
        // 使用IMU进行预测
        virtual bool PredictByImu(ImuData const& last_imu, ImuData const& cur_imu, State &state, Eigen::Vector3d const& gravity,
                          double const& acc_noise, double const& gyro_noise, double const& acc_bias_noise, double const& gyro_bias_noise)  = 0; 

        // 使用轮速进行预测
        virtual bool PredictByWheels() = 0; 

        // 使用GPS数据进行校正 
        virtual bool UpdateByGps(GpsPositionData const& gps_data, State &state) = 0; 

        // 使用激光进行校正
        virtual bool UpdateByLidar() = 0; 

    };
}


#endif


















