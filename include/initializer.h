#ifndef _INITIALIZER_H_
#define _INITIALIZER_H_

#include <queue>
#include <glog/logging.h>

#include "base_type.h"
#include "utility.hpp"

namespace MsfLocalization {

// 声明常量   constexpr 是指常量表达式 在编译过程完成计算  
const int kImuDataBufferLength = 100;
const double kAccStdLimit         = 0.1;

// 系统初始化类  
class Initializer 
{
public:
    Initializer();
    
    // 组合导航初始化 
    bool InsInitialize(GpsPositionData const& gps_data,  std::queue<ImuData> &imu_data, State &state, ImuData &last_imu);
    
private:
    // 通过加速度计算pitch和roll 
    bool compute_R_FromImuData(Eigen::Matrix3d &matrix_rot, std::vector<ImuData> const& imu_data);

    std::deque<ImuDataPtr> imu_buffer_;
};

}  // namespace MsfLocalization   
#endif