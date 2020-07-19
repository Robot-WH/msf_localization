
#ifndef UKF_HPP
#define UKF_HPP

#include <random>
#include <Eigen/Dense>

/**
 * @brief Unscented Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template<typename T, class System>
class UnscentedKalmanFilter {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
public:
  /**
   * @brief constructor          UKF的初始构造函数
   * @param system               system to be estimated  
   * @param state_dim            state vector dimension  状态维度  16
   * @param input_dim            input vector dimension  输入数据维度  6   加速度   角速度
   * @param measurement_dim      measurement vector dimension  测量向量维度  7   
   * @param process_noise        process noise covariance (state_dim x state_dim)  状态噪声方差
   * @param control_noise        控制噪声
   * @param measurement_noise    measurement noise covariance (measurement_dim x measuremend_dim)   测量噪声方差
   * @param mean                 initial mean          状态初始均值
   * @param cov                  initial covariance    状态初始方差   
   */
  UnscentedKalmanFilter(const System& system, int state_dim, int input_dim, int measurement_dim, const MatrixXt& process_noise, const MatrixXt& control_noise,const MatrixXt& measurement_noise, const VectorXt& mean, const MatrixXt& cov)
    : state_dim(state_dim),             // 16 
    input_dim(input_dim),               // 6
    measurement_dim(measurement_dim),   // 7
    N(state_dim),                       // 状态维度 16           
    M(input_dim),                       // 输入维度 6    
    K(measurement_dim),                 // 7  
    S(2 * state_dim + 1),               // sigma_point 个数   2n+1
    mean(mean),                         // 初始状态 均值
    cov(cov),                           // 初始状态 协方差
    system(system),                     // 系统的方程
    process_noise(process_noise),       // 过程噪声
    control_noise(control_noise),   
    measurement_noise(measurement_noise),   // 测量噪声
    lambda(1),                              // 超参数lamda设置为1  
    normal_dist(0.0, 1.0)
  {
    weights.resize(S, 1);           // 每个sigma点的权重
    sigma_points.resize(S, N);      // sigma 点 每行16维   
    ext_weights.resize(2 * (N + K) + 1, 1);
    ext_weights_pre.resize(2 * (N + M) + 1, 1);
    markvo_weights.resize(2 * (N + M + K) + 1, 1);
    ext_sigma_points_pre.resize(2 * (N + M) + 1, N + M);            // 机器人学状态估计方法 预测时采样
    ext_sigma_points.resize(2 * (N + K) + 1, N + K);                // hdl增广模式下 测量采样
    markvo_sigma_points.resize(2 * (N + M + K) + 1, N + M + K);
    expected_measurements.resize(2 * (N + K) + 1, K);               // hdl原装 测量
    

    // 权重初始化   
    // initialize weights for unscented filter
    weights[0] = lambda / (N + lambda);
    for (int i = 1; i < 2 * N + 1; i++) {
      weights[i] = 1 / (2 * (N + lambda));
    }

    // weights for extended state space which includes error variances
    // hdl 在测量更新时采用的权重
    ext_weights[0] = lambda / (N + K + lambda);
    for (int i = 1; i < 2 * (N + K) + 1; i++) {
      ext_weights[i] = 1 / (2 * (N + K + lambda));
    }
    // 预测时增广权重  
    ext_weights_pre[0] = lambda / (N + M + lambda);
    for (int i = 1; i < 2 * (N + M) + 1; i++) {
      ext_weights_pre[i] = 1 / (2 * (N + M + lambda));
    }
    // markvo 定位
    markvo_weights[0] = lambda / (N + M + K + lambda);
    for (int i = 1; i < 2 * (N + M +K) + 1; i++) {
      markvo_weights[i] = 1 / (2 * (N + M + K + lambda));
    }

  }

  /****************************************************概率机器人程序3.4 简单UKF的实现***********************************************/
  
  /**
   * @brief predict
   * @param control  input vector
   */
  VectorXt predict_simple(const VectorXt& control) {
    // calculate sigma points
    ensurePositiveFinite(cov);                      // 使状态的协方差矩阵正定
    computeSigmaPoints(mean, cov, sigma_points);    // 计算sigma 点
    // 对每个sigma点进行状态传播
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i), control);
    }
    // 过程噪声
    const auto& R = process_noise;
    
    // unscented transform
    // 计算预测之后的sigma点均值和方差
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    // 预测均值    对每个sigma point进行加权      
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    // 预测方差
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    // 直接叠加上噪声
    cov_pred += R;
    // 获得预测状态的均值与协方差    
    mean = mean_pred;
    cov = cov_pred;
    return mean;
  }


   /**
   * @brief correct   参考概率机器人  程序3.4
   * @param measurement  measurement vector
   */
  VectorXt correct_simple(const VectorXt& measurement) {

    // 保证正定性
    ensurePositiveFinite(cov);
    MatrixXt sigma_points_pre(S,N);                      // 用于观测的sigma_point 数量为S  维度为状态维度N
    // 采样sigma点
    computeSigmaPoints(mean, cov, sigma_points_pre);
    
    /*******************************计算预测的观测值***********************************/
    // unscented transform
    MatrixXt expected_measurements(S,K);                 // 准备观测矩阵  行数为sigma点数  列数为观测维度 7
    expected_measurements.setZero();
    // 遍历每个sigma point 对每个点获取观测值
    for (int i = 0; i < sigma_points_pre.rows(); i++) {
      // 第i个sigma 点 进行观测
      expected_measurements.row(i) = system.h(sigma_points_pre.row(i));
    }
    // 加权计算观测量
    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    for (int i = 0; i < sigma_points_pre.rows(); i++) {
      expected_measurement_mean += weights[i] * expected_measurements.row(i).transpose();
    }
    // 计算观测的不确定性
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < sigma_points_pre.rows(); i++) {
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += weights[i] * diff * diff.transpose();
    }
    // 观测噪声直接叠加
    expected_measurement_cov = expected_measurement_cov + measurement_noise;   

    // calculated transformed covariance   计算互协方差矩阵
    MatrixXt sigma = MatrixXt::Zero(N , K);
    for (int i = 0; i < sigma_points_pre.rows(); i++) {
      // N × 1  
      auto diffA = (sigma_points_pre.row(i).transpose() - mean);
      // K × 1  
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
      sigma += weights[i] * (diffA * diffB.transpose());
    }
    // 计算卡尔曼增益    互协方差矩阵 * 观测不确定性矩阵  
    kalman_gain = sigma * expected_measurement_cov.inverse();
    const auto& K = kalman_gain;
    // 计算校正后的均值和协方差
    VectorXt cor_mean = mean + K * (measurement - expected_measurement_mean);
    MatrixXt cor_cov = cov - K * expected_measurement_cov * K.transpose();
    // 获得校正后的
    mean = cor_mean;
    cov = cor_cov;
    return mean;
  }

  
   /////////////////////////////////// hdl_localization 的原装UKF校正   
  /**
   * @brief correct
   * @param measurement  measurement vector
   */
  VectorXt correct(const VectorXt& measurement) {
    // create extended state space which includes error variances   包含测量
    VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);       // 扩展的包括测量的状态
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);    // 扩展的协方差矩阵
    ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);      
    ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);     
    ext_cov_pred.bottomRightCorner(K, K) = measurement_noise;

    /******************************对预测状态采样sigma点*******************************/
    // 保证正定性
    ensurePositiveFinite(ext_cov_pred);
    // 采样sigma点
    computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points);
    
    /*******************************计算预测的观测值***********************************/
    // unscented transform
    expected_measurements.setZero();
    // 遍历每个sigma point 对每个点获取观测值
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      // 第i个sigma 点 进行观测
      expected_measurements.row(i) = system.h(ext_sigma_points.row(i).transpose().topLeftCorner(N, 1));
      // 加上噪声分量   
      expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(K, 1));
    }
    // 加权计算观测量
    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
    }
    // 计算观测的不确定性
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
    }

    // calculated transformed covariance   计算互协方差矩阵
    MatrixXt sigma = MatrixXt::Zero(N , K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      // N+K × 1  
   //   auto diffA = (ext_sigma_points.row(i).transpose() - ext_mean_pred);
      auto diffA = (ext_sigma_points.row(i).transpose().topLeftCorner(N, 1) - mean);
      // K  
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
      sigma += ext_weights[i] * (diffA * diffB.transpose());
    }
    // 计算卡尔曼增益    互协方差矩阵 * 观测不确定性矩阵  
    kalman_gain = sigma * expected_measurement_cov.inverse();    // N × k
    const auto& K = kalman_gain;
    // 计算校正后的均值和协方差
    VectorXt ext_mean = mean + K * (measurement - expected_measurement_mean);
    MatrixXt ext_cov = cov - K * expected_measurement_cov * K.transpose();

    mean = ext_mean;
    cov = ext_cov;
    return mean;
  }

  //////////////////////////////////////////////////////基于机器人学的状态估计 ukf做法  预测时加入控制噪声    未调试成功 !!!!!!!!!!!!!!!!!!!!!!!!!!  
    /**
   * @brief predict   不把噪声直接加性叠加到协方差上  而是考虑噪声对估计过程的影响       
   * @param control  input vector
   */
  VectorXt predict_ext(const VectorXt& control) {

    VectorXt ext_mean = VectorXt::Zero(N + M, 1);       // 状态+控制的噪声
    MatrixXt ext_cov = MatrixXt::Zero(N + M, N + M);    // 扩展的协方差矩阵
    ext_mean.topLeftCorner(N, 1) = VectorXt(mean);      // 先放入均值   噪声均值为0   
    ext_cov.topLeftCorner(N, N) = MatrixXt(cov);        // 设置状态噪声
    ext_cov.bottomRightCorner(M, M) = control_noise;    // 设置控制噪声

    // calculate sigma points  采样sigma点    每一行为一个sigma点 有2 * (N + M) + 1行   每行包括状态分量与控制噪声分量
    ensurePositiveFinite(ext_cov);                                  // 使状态的协方差矩阵正定
    computeSigmaPoints(ext_mean, ext_cov, ext_sigma_points_pre);    // 计算sigma 点
    // 对每个sigma点进行状态传播
    for (int i = 0; i < 2*(N+M)+1; i++) {
      VectorXt control_addnoise = control; //+ VectorXt(ext_sigma_points_pre.row(i).transpose().bottomRightCorner(M, 1));    // 控制叠加上噪声    
      ext_sigma_points_pre.row(i).transpose().topLeftCorner(N, 1) = system.f(ext_sigma_points_pre.row(i).transpose().topLeftCorner(N, 1), control_addnoise);                            // 进行状态传播   
    }
  
    // unscented transform
    // 计算预测之后的sigma点均值和方差
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    // 预测均值       
    for (int i = 0; i < 2*(N+M)+1; i++) {
      mean_pred += ext_weights_pre[i] * ext_sigma_points_pre.row(i).transpose().topLeftCorner(N, 1);
    }
    // 预测方差
    for (int i = 0; i < 2*(N+M)+1; i++) {
      VectorXt diff = ext_sigma_points_pre.row(i).transpose().topLeftCorner(N, 1) - mean_pred;
      cov_pred += ext_weights_pre[i] * diff * diff.transpose();
    }

    // 获得预测状态的均值与协方差    
    mean = mean_pred;
    cov = cov_pred;

    return mean;
  }



  /*************************************************概率机器人 马尔可夫 UKF定位 程序7.4 实现 ***************************************************/
      /**
   * @brief predict   不把噪声直接加性叠加到协方差上  而是考虑噪声对估计过程的影响       
   * @param control  input vector
   */
  VectorXt predict_markvo(const VectorXt& control) {

    VectorXt ext_mean = VectorXt::Zero(N + M + K, 1);       // 状态+控制的噪声
    MatrixXt ext_cov = MatrixXt::Zero(N + M + K, N + M + K);    // 扩展的协方差矩阵
    ext_mean.topLeftCorner(N, 1) = VectorXt(mean);      // 先放入均值   噪声均值为0   
    ext_cov.topLeftCorner(N, N) = MatrixXt(cov);        // 设置状态噪声
    ext_cov.block(N,N,M,M) = control_noise;     
    ext_cov.bottomRightCorner(K, K) = measurement_noise;    // 设置测量噪声

    // calculate sigma points  采样sigma点    每一行为一个sigma点 有2 * (N + M) + 1行   每行包括状态分量与控制噪声分量
    ensurePositiveFinite(ext_cov);                      // 使状态的协方差矩阵正定
    computeSigmaPoints(ext_mean, ext_cov, markvo_sigma_points);    // 计算sigma 点
    // 对每个sigma点进行状态传播
    for (int i = 0; i < 2*(N+M+K)+1; i++) {
      VectorXt control_addnoise = control + VectorXt(markvo_sigma_points.row(i).transpose().middleRows(N,M));    // 控制叠加上噪声    ！！！！！！！！！！！！！！！！！！！
      markvo_sigma_points.row(i).transpose().topLeftCorner(N, 1) = system.f(markvo_sigma_points.row(i).transpose().topLeftCorner(N, 1), control_addnoise);                            // 进行状态传播   
    }
  
    // unscented transform
    // 计算预测之后的sigma点均值和方差
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    // 预测均值       
    for (int i = 0; i < 2*(N+M+K)+1; i++) {
      mean_pred += markvo_weights[i] * markvo_sigma_points.row(i).transpose().topLeftCorner(N, 1);
    }
    // 预测方差
    for (int i = 0; i < 2*(N+M+K)+1; i++) {
      VectorXt diff = markvo_sigma_points.row(i).transpose().topLeftCorner(N, 1) - mean_pred;
      cov_pred += markvo_weights[i] * diff * diff.transpose();
    }

    // 获得预测状态的均值与协方差    
    mean = mean_pred;
    cov = cov_pred;

    return mean;
  }

  /**
   * @brief correct
   * @param measurement  measurement vector
   */
  VectorXt correct_markvo(const VectorXt& measurement) {
   
    MatrixXt expected_measurements(2*(N+M+K)+1,K);                 // 准备观测矩阵  行数为sigma点数  列数为观测维度 7
    /*******************************计算预测的观测值***********************************/
    // unscented transform
    expected_measurements.setZero();
    // 遍历每个sigma point 对每个点获取观测值
    for (int i = 0; i < markvo_sigma_points.rows(); i++) {
      // 第i个sigma 点 进行观测
      expected_measurements.row(i) = system.h(markvo_sigma_points.row(i).transpose().topLeftCorner(N, 1));
      // 加上噪声分量   
      expected_measurements.row(i) += VectorXt(markvo_sigma_points.row(i).transpose().bottomRightCorner(K, 1));
    }
    // 加权计算观测量
    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    for (int i = 0; i < markvo_sigma_points.rows(); i++) {
      expected_measurement_mean += markvo_weights[i] * expected_measurements.row(i);
    }
    // 计算观测的不确定性
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < markvo_sigma_points.rows(); i++) {
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += markvo_weights[i] * diff * diff.transpose();
    }

    // calculated transformed covariance   计算互协方差矩阵
    MatrixXt sigma = MatrixXt::Zero(N, K);
    for (int i = 0; i < markvo_sigma_points.rows(); i++) {
      // N+K × 1  
      auto diffA = (markvo_sigma_points.row(i).transpose().topLeftCorner(N, 1) - mean);
      // K  
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
      sigma += markvo_weights[i] * (diffA * diffB.transpose());
    }
    // 计算卡尔曼增益    互协方差矩阵 * 观测不确定性矩阵  
    kalman_gain = sigma * expected_measurement_cov.inverse();
    const auto& K = kalman_gain;
    // 计算校正后的均值和协方差
    VectorXt ext_mean = mean + K * (measurement - expected_measurement_mean);
    MatrixXt ext_cov = cov - K * expected_measurement_cov * K.transpose();

    mean = ext_mean;
    cov = ext_cov;

    return mean;
  }
  

 

  /*			getter			*/
  const VectorXt& getMean() const { return mean; }
  const MatrixXt& getCov() const { return cov; }
  const MatrixXt& getSigmaPoints() const { return sigma_points; }

  System& getSystem() { return system; }
  const System& getSystem() const { return system; }
  const MatrixXt& getProcessNoiseCov() const { return process_noise; }
  const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise; }

  const MatrixXt& getKalmanGain() const { return kalman_gain; }

  /*			setter			*/
  UnscentedKalmanFilter& setMean(const VectorXt& m) { mean = m;			return *this; }
  UnscentedKalmanFilter& setCov(const MatrixXt& s) { cov = s;			return *this; }

  UnscentedKalmanFilter& setProcessNoiseCov(const MatrixXt& p) { process_noise = p;			return *this; }
  UnscentedKalmanFilter& setMeasurementNoiseCov(const MatrixXt& m) { measurement_noise = m;	return *this; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const int state_dim;
  const int input_dim;
  const int measurement_dim;

  const int N;
  const int M;
  const int K;
  const int S;

public:
  VectorXt mean;
  MatrixXt cov;

  System system;
  MatrixXt process_noise;		// 
  MatrixXt control_noise;    
  MatrixXt measurement_noise;	//

  T lambda;
  VectorXt weights;

  MatrixXt sigma_points;
  MatrixXt ext_sigma_points;
  VectorXt markvo_weights;
  VectorXt ext_weights;
  VectorXt ext_weights_pre;           // 增广状态预测时的权重
  MatrixXt ext_sigma_points_pre;
  MatrixXt markvo_sigma_points;
  MatrixXt expected_measurements;

private:
  /**
   * @brief compute sigma points    根据当前状态采样sigma点
   * @param mean          mean
   * @param cov           covariance
   * @param sigma_points  calculated sigma points    sigma点的矩阵 第i行是第i个sigma点的状态均值
   */
  void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points) {
    const int n = mean.size();                      // 状态的维度
    assert(cov.rows() == n && cov.cols() == n);    
    // 计算sqrt((n+lamda)cov)   进行cholesky分解  A = LLt  然后取L
    Eigen::LLT<MatrixXt> llt; 
    llt.compute((n + lambda) * cov);
    MatrixXt l = llt.matrixL();
    
    sigma_points.row(0) = mean;                     // 第0个sigma点赋值为均值
    // 每个状态维度 产生两个sigma点
    for (int i = 0; i < n; i++) {
      sigma_points.row(1 + i * 2) = mean + l.col(i);
      sigma_points.row(1 + i * 2 + 1) = mean - l.col(i);
    }
  }

  /**
   * @brief make covariance matrix positive finite   保证矩阵的正定性   因为在采样sigma点时 需要对矩阵求cholesky分解 所以要保证正定
   * @param cov  covariance matrix
   */
  void ensurePositiveFinite(MatrixXt& cov) {
    return;
    const double eps = 1e-9;
    
    Eigen::EigenSolver<MatrixXt> solver(cov);
    MatrixXt D = solver.pseudoEigenvalueMatrix();        // 求矩阵的特征值对角阵    
    MatrixXt V = solver.pseudoEigenvectors();            // 求特征向量矩阵
    // 遍历特征值对角阵将小于eps的手动设置为eps
    for (int i = 0; i < D.rows(); i++) {
      if (D(i, i) < eps) {
        D(i, i) = eps;
      }
    }
    // 最后恢复原矩阵  保证正定
    cov = V * D * V.inverse();
  }


public:
  MatrixXt kalman_gain;

  std::mt19937 mt;
  std::normal_distribution<T> normal_dist;
};


#endif
