
#ifndef EKF_HPP
#define EKF_HPP


#include <Eigen/Dense>

using namespace Eigen;

 template<typename T> 
 class ExtendedKalmanFilter
 {
 public:
     double dt;  

     typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
     typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
     typedef Eigen::Quaternion<T> Quaterniont;
     typedef Eigen::Matrix<T, 3, 1> Vector3t;

     ExtendedKalmanFilter(const VectorXt& mean, const MatrixXt& cov):mean(mean),cov(cov)
     {
       dt = 0;
       // 预测过程噪声     非加性   
       predict_noise = MatrixXt::Identity(12, 12);
       predict_noise.middleRows(0, 3) *= 0.5;       // 加速度的噪声
       predict_noise.middleRows(3, 3) *= 0.1;       // 角速度噪声
       predict_noise.middleRows(6, 3) *= 1e-6;      // 偏置噪声
       predict_noise.middleRows(9, 3) *= 1e-6;      
       // 测量噪声     加性
       measurement_noise = MatrixXt::Identity(7, 7);
       measurement_noise.middleRows(0, 3) *= 0.01;
       measurement_noise.middleRows(3, 4) *= 0.001;
       
       // 观测矩阵初始化     观测位置 旋转 
       H.resize(7,16);   
       H << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;


       // jacobian 初始化
       Jacobian_X = MatrixXt::Zero(16, 16);
       Jacobian_X << 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    
      Jacobian_noise = MatrixXt::Zero(16, 12);
      Jacobian_noise << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0,  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                        0,  0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                        0,  0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                        0,  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                        0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                        0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
     
     
     }

     virtual ~ExtendedKalmanFilter() {}
  
  //反对称矩阵
  MatrixXt Skewsymmetric(VectorXt v)
  {
     MatrixXt m = MatrixXt::Zero(3, 3);
     m << 0, -v[3], v[2],
         v[3],  0, -v[1],
        -v[2], v[1],  0;
     return m;
  }  

  // 设置状态转移方程关于状态的jacobian
  void Jacobian_get(const VectorXt& control,const VectorXt& state)
  {
    // 状态的jacobian
    // 控制量
    Vector3t raw_acc = control.middleRows(0, 3);               // 加速度计               
    Vector3t raw_gyro = control.middleRows(3, 3);              // 角速度
    Jacobian_X(0,3) = dt;
    Jacobian_X(1,4) = dt;
    Jacobian_X(2,5) = dt;

    // 姿态四元数
    Vector3t gyro_bias = state.middleRows(13, 3);              // 陀螺仪偏置
//    Jacobian_X.block<1,3>(6,7) = -((raw_gyro - gyro_bias)*dt).transpose()*0.5;
//    Jacobian_X.block<3,1>(7,6) = (raw_gyro - gyro_bias)*dt*0.5;
//    Jacobian_X.block<3,3>(7,7) = (MatrixXt::Identity(3,3) - Skewsymmetric((raw_gyro - gyro_bias)*dt))*0.5; 
    Jacobian_X.block(6,7,1,3) = -((raw_gyro - gyro_bias)*dt).transpose()*0.5;
    Jacobian_X.block(7,6,3,1) = (raw_gyro - gyro_bias)*dt*0.5;
    Jacobian_X.block(7,7,3,3) = 0.5*(MatrixXt::Identity(3,3) - Skewsymmetric((raw_gyro - gyro_bias)*dt)); 

    // 噪声jacobian 
    Jacobian_noise(0,0) = dt;
    Jacobian_noise(1,1) = dt;  
    Jacobian_noise(2,2) = dt;

    Jacobian_noise.block(6,3,1,3) = -state.middleRows(7, 3).transpose()*MatrixXt::Identity(3,3)*dt*0.5;
    Jacobian_noise.block(7,3,3,3) = state[6]*0.5*dt*MatrixXt::Identity(3,3) + Skewsymmetric(state.middleRows(7, 3))*MatrixXt::Identity(3,3)*dt*0.5;
    
  }


  // 系统状态转移方程  
  // state：当前状态    pt:(X Y Z)   vt(vx vy vz)  Q(q1 q2 q3 q4)  acc_bias   gyro_bias    
  // control: 控制   
  VectorXt F_x_u(const VectorXt& state, const VectorXt& control) const {
    
    VectorXt next_state(16);
    // 先读取 t-1时刻状态
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
    
 // 预测
 VectorXt predict(const VectorXt& control)
 {
   // 更新jacobian
   Jacobian_get(control,mean);
   // 预测的均值
   mean = F_x_u(mean, control);
   // 预测的方差
   cov = Jacobian_X*cov*Jacobian_X.transpose() + Jacobian_noise*predict_noise*Jacobian_noise.transpose();
   return mean;
 }


VectorXt correct(const VectorXt& measurement)
{
  // 卡尔曼增益     16*7
  MatrixXt K = (H * cov * H.transpose() + measurement_noise).inverse();
  K = cov*H.transpose()*K;
  mean = mean + K * (measurement - H*mean);
  cov = (MatrixXt::Identity(16,16) - K*H)*cov;
  return mean;   
}

void setProcessNoiseCov(double dt)
{
   predict_noise = predict_noise*dt;
}

 
 private:
     /* data */
      
     // 状态的均值与协方差
     VectorXt mean;    
     MatrixXt cov;   
     // 状态转移方程关于状态的jacobian
     MatrixXt Jacobian_X;
     MatrixXt Jacobian_noise;
     MatrixXt predict_noise;    
     // 观测矩阵
     MatrixXt H;
     MatrixXt measurement_noise;    
 
 };
 




#endif