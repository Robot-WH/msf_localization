#ifndef ESKF_H
#define ESKF_H

#include "filter.hpp"

namespace MsfLocalization{

// eskf 融合的实现  
class eskf : public filter
{
    public:
        eskf()
        {
        }

        ~eskf(){}

        /**
         * 
         * @brief 对于IMU进行预测
         * @param last_imu 上一刻用于预测的imu数据  如果上一刻采用了GPS校正  则需要插值出GPS时刻处IMU数据作为last_imu
         * @param cur_imu 当前时刻imu数据 
         * @param state 当前维护的状态 
         * @param gravity 重力数值
         * @param acc_noise 加速度测量噪声
         * @param gyro_noise 陀螺仪测量噪声 
         * @param acc_bias_noise 加速度偏置随机游走噪声
         * @param gyro_bias_noise 陀螺仪偏置随机游走噪声 
         */
        bool PredictByImu(ImuData const& last_imu, ImuData const& cur_imu, State &state, Eigen::Vector3d const& gravity,
                          double const& acc_noise, double const& gyro_noise, double const& acc_bias_noise, double const& gyro_bias_noise) 
        {
            // Time.   两个IMU的时间差    
            const double delta_t = cur_imu.timestamp - last_imu.timestamp;
            const double delta_t2 = delta_t * delta_t;

            // Set last state.
            State last_state = state;

            // ***************** 采用中值积分进行预测 ***************************
            // 陀螺仪中值  
            const Eigen::Vector3d gyro_unbias = 0.5 * (last_imu.gyro + cur_imu.gyro) - last_state.gyro_bias;
            const Eigen::Vector3d delta_angle_axis = gyro_unbias * delta_t;     // (wm - wb)*dt  
            // 首先更新当前时刻旋转  
            if (delta_angle_axis.norm() > 1e-12) 
            {
                state.R = last_state.R * Eigen::AngleAxisd(delta_angle_axis.norm(), delta_angle_axis.normalized()).toRotationMatrix();
            }
            // 中值加速度
            const Eigen::Vector3d acc_unbias = 
                        0.5 * ( last_state.R * (last_imu.acc - last_state.acc_bias) + state.R * (cur_imu.acc - last_state.acc_bias));
            // 速度 
            state.V = last_state.V + (acc_unbias + gravity) * delta_t;
            // 位移 
            state.P = last_state.P + last_state.V * delta_t + 
                        0.5 * (acc_unbias + gravity) * delta_t2;
            
            // error state 的预测均值为0   

            // Covariance of the error-state.   协方差
            // Fx  error_state的转移矩阵       15x15  
            Eigen::Matrix<double, 15, 15> Fx = Eigen::Matrix<double, 15, 15>::Identity();
            Fx.block<3, 3>(0, 3)   = Eigen::Matrix3d::Identity() * delta_t;
            Fx.block<3, 3>(6, 12)  = - Eigen::Matrix3d::Identity() * delta_t;
            Fx.block<3, 3>(3, 6)   = - last_state.R * GetSkewMatrix(last_imu.acc - last_state.acc_bias) * delta_t;
            Fx.block<3, 3>(3, 9)   = - last_state.R * delta_t;
            if (delta_angle_axis.norm() > 1e-12) 
            {   // 轴角 -> 旋转矩阵   
                Fx.block<3, 3>(6, 6) = Eigen::AngleAxisd(delta_angle_axis.norm(), delta_angle_axis.normalized()).toRotationMatrix().transpose();
            }else 
            {
                Fx.block<3, 3>(6, 6).setIdentity();
            }
            
            // Fi   15x12
            Eigen::Matrix<double, 15, 12> Fi = Eigen::Matrix<double, 15, 12>::Zero();
            Fi.block<12, 12>(3, 0) = Eigen::Matrix<double, 12, 12>::Identity();
            // Qi   12x12
            Eigen::Matrix<double, 12, 12> Qi = Eigen::Matrix<double, 12, 12>::Zero();
            Qi.block<3, 3>(0, 0) = delta_t2 * acc_noise * Eigen::Matrix3d::Identity();     // t^2 * an
            Qi.block<3, 3>(3, 3) = delta_t2 * gyro_noise * Eigen::Matrix3d::Identity();    // t^2 * wn 
            Qi.block<3, 3>(6, 6) = delta_t * acc_bias_noise * Eigen::Matrix3d::Identity(); // t * nba
            Qi.block<3, 3>(9, 9) = delta_t * gyro_bias_noise * Eigen::Matrix3d::Identity();// t * nbw
            // 求得IMU预测量的 协方差    15x15
            state.cov = Fx * last_state.cov * Fx.transpose() + Fi * Qi * Fi.transpose();

            // Time and imu.
            state.timestamp = cur_imu.timestamp;
            // state.imu_data_ptr = cur_imu;
        }

        // 对于轮速进行预测
        bool PredictByWheels()
        {

        }

        /**
         * @brief 对于GPS进行校正
         * @param gps_data gps 观测值   注意这个观测值要转换到state相同的坐标系上 
         */
        bool UpdateByGps(GpsPositionData const& gps_data, State &state)
        {   
            Eigen::Matrix<double, 3, 15> H;               // GPS的观测矩阵 
            // 计算残差 
            Eigen::Vector3d residual;
            // GPS残差   gps的观测值 - 预测值    3x1 
            residual = gps_data.enu - state.P;            
            computeJacobianOfGPS(H, state.R);             // 计算观测的jacobian H 
            const Eigen::Matrix3d& V = gps_data.cov;      // 观测噪声   3x3   
            const Eigen::MatrixXd& P = state.cov;         // 预测量的 协方差    15x15
            // 15 x 3
            const Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + V).inverse();  // 卡尔曼增益  和预测协方差P与观测协方差 V 有关 
            // 15 x 1
            const Eigen::VectorXd delta_x = K * residual;    // 计算error state   注意 预测状态的均值一直为0    所以不需要+error_state

            // Add delta_x to state.   将error state更新到状态中
            addErrorToNominal(delta_x, state);

            // Covarance.   更新协方差    
            const Eigen::MatrixXd I_KH = Eigen::Matrix<double, 15, 15>::Identity() - K * H;
            state.cov = I_KH * P * I_KH.transpose() + K * V * K.transpose();    
        }

        // 对于Lidar进行校正 
        bool UpdateByLidar()
        {

        }


    private: 
        
        // 对于GPS的观测  计算观测jacobian与观测残差 
        // GPS的观测为 XYZ
        void computeJacobianOfGPS(Eigen::Matrix<double, 3, 15> &jacobian, Eigen::Matrix3d const& pose)
        {
            Eigen::Matrix<double, 3, 16> dev_x = Eigen::Matrix<double, 3, 16>::Zero();
            dev_x.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

            Eigen::Matrix<double, 16, 15> dev_error_x = Eigen::Matrix<double, 16, 15>::Zero();
            dev_error_x.block<6,6>(0,0) = Eigen::Matrix<double, 6, 6>::Identity(); 
            // 将matrix转为四元数   
            Eigen::Quaterniond q(pose);
            dev_error_x.block<4, 3>(6, 6) << 0.5 * -q.x(), 0.5 * -q.y(), 0.5 * -q.z(),
                0.5 * q.w(), 0.5 * -q.z(), 0.5 * q.y(),
                0.5 * q.z(), 0.5 * q.w(), 0.5 * -q.x(),
                0.5 * -q.y(), 0.5 * q.x(), 0.5 * q.w();
            dev_error_x.block<6, 6>(10, 9) = Eigen::Matrix<double, 6, 6>::Identity();

            jacobian = dev_x * dev_error_x;  
        }

        // 对于Lidar的观测  计算观测jacobian与观测残差 
        // Lidar的观测为 XYZ+RPY 
        void ComputeJacobianOfLidar()
        {

        }

        // 将error state injection 到nominal state中   
        void addErrorToNominal(Eigen::Matrix<double, 15, 1> const& delta_x, State &state) 
        {
            state.P     += delta_x.block<3, 1>(0, 0);
            state.V     += delta_x.block<3, 1>(3, 0);
            state.acc_bias  += delta_x.block<3, 1>(9, 0);
            state.gyro_bias += delta_x.block<3, 1>(12, 0);
            // 更新旋转  delta_x 是旋转向量   先要转换到旋转矩阵 
            if (delta_x.block<3, 1>(6, 0).norm() > 1e-12) 
            {
                state.R *= Eigen::AngleAxisd(delta_x.block<3, 1>(6, 0).norm(), delta_x.block<3, 1>(6, 0).normalized()).toRotationMatrix();
            }
        }

};

}

#endif