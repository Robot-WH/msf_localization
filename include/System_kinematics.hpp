#ifndef SYSTEM_KINEMATICS_HPP
#define SYSTEM_KINEMATICS_HPP

#include <filters/ukf.hpp>


/**
 * @brief Definition of system to be estimated by ukf
 * @note state = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
 */
class PoseSystem {
public:
  typedef float T;
  typedef Eigen::Matrix<T, 3, 1> Vector3t;
  typedef Eigen::Matrix<T, 4, 4> Matrix4t;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Quaternion<T> Quaterniont;
public:
  PoseSystem() {
    dt = 0.01;
  }

  // system equation
  // state：当前状态      control: 控制   
  VectorXt f(const VectorXt& state, const VectorXt& control) const {
    VectorXt next_state(16);
    // 先读取状态  
    Vector3t pt = state.middleRows(0, 3);     // 位置
    Vector3t vt = state.middleRows(3, 3);     // 速度
    Quaterniont qt(state[6], state[7], state[8], state[9]);    // 四元数
    qt.normalize();                                            // 四元数归一化    

    Vector3t acc_bias = state.middleRows(10, 3);               // 加速度偏置
    Vector3t gyro_bias = state.middleRows(13, 3);              // 陀螺仪偏置
    // 控制量
    Vector3t raw_acc = control.middleRows(0, 3);               // 加速度计               
    Vector3t raw_gyro = control.middleRows(3, 3);              // 角速度
    
    // 下面要根据上一时刻的状态量  以及当前的控制输入  预测当前的状态量
    // position
    next_state.middleRows(0, 3) = pt + vt * dt;					       // 恒速运动模型  更新当前位置P   

    // velocity
    Vector3t g(0.0f, 0.0f, -9.80665f);
    Vector3t acc_ = raw_acc - acc_bias;                        // 去除偏置
    Vector3t acc = qt * acc_;                                  // 加速度转移到世界坐标系  
    // 认为是匀速运动
    next_state.middleRows(3, 3) = vt;                       // + (acc - g) * dt;		// acceleration didn't contribute to accuracy due to large noise

    // orientation
    Vector3t gyro = raw_gyro - gyro_bias;                                           // 减去偏置
    Quaterniont dq(1, gyro[0] * dt / 2, gyro[1] * dt / 2, gyro[2] * dt / 2);        // 该旋转转换到四元数   注意是小量
    dq.normalize();                                                                 // 单位化    旋转的四元数要记得单位化
    Quaterniont qt_ = (qt * dq).normalized();                                       // 四元数的更新
    next_state.middleRows(6, 4) << qt_.w(), qt_.x(), qt_.y(), qt_.z();
    // 对于观测量nominal 认为偏置是恒定的   没有随机游走
    next_state.middleRows(10, 3) = state.middleRows(10, 3);		// constant bias on acceleration
    next_state.middleRows(13, 3) = state.middleRows(13, 3);		// constant bias on angular velocity

    return next_state;
  }

  // observation equation    观测模型   
  VectorXt h(const VectorXt& state) const {
    VectorXt observation(7);
    observation.middleRows(0, 3) = state.middleRows(0, 3);
    observation.middleRows(3, 4) = state.middleRows(6, 4).normalized();

    return observation;
  }

  double dt;
};


#endif // POSE_SYSTEM_HPP
