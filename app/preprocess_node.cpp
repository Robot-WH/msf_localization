#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
// 表示障碍物的box   
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <opencv/cv.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/features/normal_3d.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/filters/impl/plane_clipper3D.hpp>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <pcl/segmentation/extract_clusters.h>

#include "omp.h"

/****
 *    数据预处理模块:
 *    1 点云滤波
 *    2 地面去除
 *    3 去畸变
 **/

using namespace std;

typedef pcl::PointXYZI PointT;

ros::Subscriber laser_sub;
ros::Subscriber imu_sub;
ros::Publisher points_pub;  
ros::Publisher floor_pub;
ros::Publisher pub_bounding_boxs;  

pcl::Filter<PointT>::Ptr downsample_filter;         // 降采样滤波器对象
pcl::Filter<PointT>::Ptr outlier_removal_filter;    // 离群点滤波对象


/*********** 全局参数  *********************/
bool floor_remove_enable = false;   // 去地面 

float distance_near_thresh =0;
float distance_far_thresh =0;

// 法向量阈值   
double normal_filter_thresh;
// 高度滤波 
float clip_height;  
int floor_pts_thresh = 500; 

// 距离图像  
cv::Mat rangeMat;
cv::Mat labelMat;       // 类别图像   
pcl::PointCloud<PointT>::Ptr fullCloud;
std::vector<std::pair<uint8_t, uint8_t> > neighborIterator;


#define use_fast_segmentation 0
#define N_SCANS 32

// 雷达模型 最底层激光scan的角度  
const float ang_bottom = 15.0 + 0.1;
const float ang_res_y = 2.0;
const float segmentTheta = 1.0472;     
const int segmentValidPointNum = 5;      
const int segmentValidLineNum = 8;      // 竖直方向聚类数量大于该值则认为是个聚类  
const float sensorMountAngle = 0.0;     // 雷达的倾角  


extern const int Horizon_SCAN = 1800;
extern const float ang_res_x = 0.2;

int labelCount = 1;    // 聚类的label   

// 用于BFS的队列    
vector<uint16_t> queueIndX(N_SCANS*Horizon_SCAN);
vector<uint16_t> queueIndY(N_SCANS*Horizon_SCAN);
// 保存当前聚类的点 
vector<uint16_t> allPushedIndX(N_SCANS*Horizon_SCAN);
vector<uint16_t> allPushedIndY(N_SCANS*Horizon_SCAN);


#if N_SCANS==16
float segmentAlphaX = 2*M_PI / Horizon_SCAN;
float segmentAlphaY = 30 / (16-1);
#elif N_SCANS==32
float segmentAlphaX = 2*M_PI / Horizon_SCAN;
float segmentAlphaY = 40 / (32-1);
#elif N_SCANS==64
float segmentAlphaX = 2*M_PI / Horizon_SCAN;
float segmentAlphaY = 26.8 / (64-1);
#endif


// 距离滤波 
pcl::PointCloud<PointT>::Ptr distance_filter(const pcl::PointCloud<PointT>::ConstPtr& cloud){
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    filtered->reserve(cloud->size());
    std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points),
        [&](const PointT& p) {                     // 外部变量按引用传递   
        double d = p.getVector3fMap().norm();
        return d > distance_near_thresh && d < distance_far_thresh;
        }
    );
    filtered->width = filtered->size();       // 点云的数量
    filtered->height = 1;
    filtered->is_dense = false;
    filtered->header = cloud->header;
    return filtered;
}

// 降采样滤波
pcl::PointCloud<PointT>::Ptr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud){
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;
    return filtered;
}


// 离群点去除
pcl::PointCloud<PointT>::Ptr outlier_removal(const pcl::PointCloud<PointT>::ConstPtr& cloud){
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    outlier_removal_filter->setInputCloud(cloud);
    outlier_removal_filter->filter(*filtered);
    filtered->header = cloud->header;
    return filtered;
}   


// 提取出一个高度以下的点云 
// out: clip_height一下的点云
// 返回剔除的点云
pcl::PointCloud<PointT> clip_above(double clip_height, const pcl::PointCloud<pcl::PointXYZI>::Ptr& in,
                             const pcl::PointCloud<pcl::PointXYZI>::Ptr& out)
{
    pcl::ExtractIndices<PointT> cliper;
    pcl::PointCloud<PointT> removal;
    // 设置要提取的输入点云
    cliper.setInputCloud(in);
    pcl::PointIndices indices;
    //omp_set_num_threads(2);
    #pragma omp for                // 没啥用 ??????
    for (size_t i = 0; i < in->points.size(); i++)
    {   // z轴高度大于 阈值   则放置到 indices   
        if (in->points[i].z > clip_height)
        {
            indices.indices.push_back(i);        // 将序序号提取出来  
            removal.push_back(in->points[i]);    // 非地面点放置与  removal 
        }
    }
    cliper.setIndices(boost::make_shared<pcl::PointIndices>(indices));
    cliper.setNegative(true);    //ture to remove the indices    剔除这部分 
    cliper.filter(*out);
    return removal;
}

/**
 * @brief filter points with non-vertical normals
 * @param cloud  input cloud
 * @return filtered cloud
 */
pcl::PointCloud<PointT>::Ptr normal_filtering(const pcl::PointCloud<PointT>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr& no_floor)  {
    // 法向量估计  
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    ne.setSearchMethod(tree);       // 设置搜索方法
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);    // 法向量点云  
    ne.setKSearch(10);
    // ne.setViewPoint(0.0f, 0.0f, sensor_height);
    ne.compute(*normals);     // 求法向量

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    filtered->reserve(cloud->size());
    //omp_set_num_threads(2);
    #pragma omp for     // 没啥效果  
    for (int i = 0; i < cloud->size(); i++) {
        // 与z轴向量点积   
        float dot = normals->at(i).getNormalVector3fMap().normalized().dot(Eigen::Vector3f::UnitZ());
        // 将夹角小于阈值的提取
        if (std::abs(dot) > std::cos(normal_filter_thresh * M_PI / 180.0)) {
          filtered->push_back(cloud->at(i));
        }
        else
        {
          no_floor->push_back(cloud->at(i));
        }   
    }
    return filtered;
}

// 地面去除  RANSAC方法
// 输入: 待去除地面的点云
// 返回去除地面的点云  
pcl::PointCloud<PointT>::Ptr floor_remove(const pcl::PointCloud<PointT>::Ptr& cloud)     // cloud 不能被修改   
{
    pcl::PointCloud<PointT>::Ptr no_floor_cloud(new pcl::PointCloud<PointT>());  // 去除了地面的点云  
    pcl::PointCloud<PointT>::Ptr floor_cloud(new pcl::PointCloud<PointT>());     //  地面的点云 
    *no_floor_cloud += clip_above(clip_height, cloud, floor_cloud);              // 高度滤波  
    *floor_cloud = *normal_filtering(floor_cloud, no_floor_cloud);              // 法线滤波
    // RANSAC提取平面
    // RANSAC 拟合平面 
    pcl::SampleConsensusModelPlane<PointT>::Ptr model_p(new pcl::SampleConsensusModelPlane<PointT>(floor_cloud));
    pcl::RandomSampleConsensus<PointT> ransac(model_p);
    ransac.setDistanceThreshold(0.5);       // 与该平面距离小于该阈值的点为内点 
    ransac.computeModel();
    // 获取内点  
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    ransac.getInliers(inliers->indices);

    // too few inliers
    if(inliers->indices.size() < floor_pts_thresh) {
      return cloud;
    }
    // 提取非地面点  
    pcl::PointCloud<PointT>::Ptr outlier_cloud(new pcl::PointCloud<PointT>);
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(floor_cloud);
    extract.setIndices(inliers);     
    extract.setNegative(true);    //ture to remove the indices    剔除这部分 
    extract.filter(*outlier_cloud);
    *no_floor_cloud += *outlier_cloud; 
    // 提取地面点
    pcl::PointCloud<PointT>::Ptr inlier_cloud(new pcl::PointCloud<PointT>);
    extract.setNegative(false);    //ture to remove the indices    剔除这部分 
    extract.filter(*inlier_cloud);
    // 发布  地面点
    sensor_msgs::PointCloud2 floor;
    pcl::toROSMsg(*inlier_cloud, floor);
    floor.header.stamp=ros::Time::now(); 
    floor.header.frame_id = cloud->header.frame_id;
    //   cout<<"frame_id: "<<output.header.frame_id<<endl;
    floor_pub.publish(floor);

    no_floor_cloud->width = no_floor_cloud->size();       // 点云的数量
    no_floor_cloud->height = 1;
    no_floor_cloud->is_dense = false;
    no_floor_cloud->header = cloud->header;
    return no_floor_cloud;
}

// 物体检测的结构体 
struct Detected_Obj
{   // jsk_recognition_msgs  这个玩意需要安装 
    jsk_recognition_msgs::BoundingBox bounding_box_;

    pcl::PointXYZ min_point_;    // box 最小的边界
    pcl::PointXYZ max_point_;    // box 最大的边界
    pcl::PointXYZ centroid_;     // box的中心坐标               
};


// 欧式聚类分割  
void cluster_segment(pcl::PointCloud<PointT>::Ptr& no_floor_cloud, int MIN_CLUSTER_SIZE, 
                    int MAX_CLUSTER_SIZE,  std_msgs::Header cloud_header_)
{
    /*   投影到2D图像的聚类
    // 首先按照距离选择聚类半径  
    //0 => 0-15m d=0.5
    //1 => 15-30 d=1
    //2 => 30-45 d=1.6
    //3 => 45-60 d=2.1
    //4 => >60   d=2.6
    // 按聚类半径分配点云 
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> segment_pc_array(5);
    vector<int> seg_distance_ = {15,30,45,60,80};
    vector<float> cluster_distance = {0.3,0.6,1,1.5,2};
    vector<Detected_Obj> obj_list;
    
    for (size_t i = 0; i < segment_pc_array.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
        segment_pc_array[i] = tmp;
    }
    // 遍历非地面所有点  
    for (size_t i = 0; i < no_floor_cloud->points.size(); i++)
    {
        pcl::PointXYZ current_point;
        current_point.x = no_floor_cloud->points[i].x;
        current_point.y = no_floor_cloud->points[i].y;
        current_point.z = no_floor_cloud->points[i].z;

        float origin_distance = sqrt(pow(current_point.x, 2) + pow(current_point.y, 2));

        // 如果点的距离大于120m, 忽略该点
        if (origin_distance >= 100)
        {
            continue;
        }

        if (origin_distance < seg_distance_[0])
        {
            segment_pc_array[0]->points.push_back(current_point);
        }
        else if (origin_distance < seg_distance_[1])
        {
            segment_pc_array[1]->points.push_back(current_point);
        }
        else if (origin_distance < seg_distance_[2])
        {
            segment_pc_array[2]->points.push_back(current_point);
        }
        else if (origin_distance < seg_distance_[3])
        {
            segment_pc_array[3]->points.push_back(current_point);
        }
        else
        {
            segment_pc_array[4]->points.push_back(current_point);
        }
    }

    // 进行聚类  
    // 创建用于搜索的kdtree    
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    
    // create 2d pc
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZ>);
    // 遍历筛选出来的每一个距离的点云 
    for(int i=0; i<5; i++)
    {   // 将该点云压缩到2D
        pcl::copyPointCloud(*segment_pc_array[i], *cloud_2d);
        // make it flat  
        /*
        for (size_t i = 0; i < cloud_2d->points.size(); i++)
        {
            cloud_2d->points[i].z = 0;
        }   
        // 设置为kdtree的
        if (cloud_2d->points.size() > 0)
            tree->setInputCloud(cloud_2d);
        // 本帧点云聚类的结果   
        std::vector<pcl::PointIndices> local_indices;
        // 搞懂聚类的原理    
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclid;
        euclid.setInputCloud(cloud_2d);
        euclid.setClusterTolerance(cluster_distance[i]);    // 设置欧式距离的buffer
        euclid.setMinClusterSize(MIN_CLUSTER_SIZE);             // 设置一个类别的最小数据点数
        euclid.setMaxClusterSize(MAX_CLUSTER_SIZE);             // 设置一个类别的最大数据点数
        euclid.setSearchMethod(tree);
        euclid.extract(local_indices);
        // 遍历所有的聚类 
        for (size_t i = 0; i < local_indices.size(); i++)
        {
            // the structure to save one detected object
            Detected_Obj obj_info;

            float min_x = std::numeric_limits<float>::max();
            float max_x = -std::numeric_limits<float>::max();
            float min_y = std::numeric_limits<float>::max();
            float max_y = -std::numeric_limits<float>::max();
            float min_z = std::numeric_limits<float>::max();
            float max_z = -std::numeric_limits<float>::max();
            // 遍历该聚类的所有点    采用索引方式  
            for (auto pit = local_indices[i].indices.begin(); pit != local_indices[i].indices.end(); ++pit)
            {
                //fill new colored cluster point by point
                pcl::PointXYZ p;
                p.x = no_floor_cloud->points[*pit].x;
                p.y = no_floor_cloud->points[*pit].y;
                p.z = no_floor_cloud->points[*pit].z;
                // 累计每个点的坐标  稍后需要求均值获取中心点  
                obj_info.centroid_.x += p.x;
                obj_info.centroid_.y += p.y;
                obj_info.centroid_.z += p.z;
                // 更新该聚类的范围  
                if (p.x < min_x)
                    min_x = p.x;
                if (p.y < min_y)
                    min_y = p.y;
                if (p.z < min_z)
                    min_z = p.z;
                if (p.x > max_x)
                    max_x = p.x;
                if (p.y > max_y)
                    max_y = p.y;
                if (p.z > max_z)
                    max_z = p.z;
            }

            // 更新聚类范围  用于绘制bounding box   
            obj_info.min_point_.x = min_x;
            obj_info.min_point_.y = min_y;
            obj_info.min_point_.z = min_z;

            obj_info.max_point_.x = max_x;
            obj_info.max_point_.y = max_y;
            obj_info.max_point_.z = max_z;

            // 计算聚类中心   
            if (local_indices[i].indices.size() > 0)
            {
                obj_info.centroid_.x /= local_indices[i].indices.size();
                obj_info.centroid_.y /= local_indices[i].indices.size();
                obj_info.centroid_.z /= local_indices[i].indices.size();
            }

            //calculate bounding box   box的size   
            double length_ = obj_info.max_point_.x - obj_info.min_point_.x;
            double width_ = obj_info.max_point_.y - obj_info.min_point_.y;
            double height_ = obj_info.max_point_.z - obj_info.min_point_.z;

            //if(length_>=5||width_>=5||height_>=5)   continue;

            obj_info.bounding_box_.header = cloud_header_;

            obj_info.bounding_box_.pose.position.x = obj_info.min_point_.x + length_ / 2;
            obj_info.bounding_box_.pose.position.y = obj_info.min_point_.y + width_ / 2;
            obj_info.bounding_box_.pose.position.z = obj_info.min_point_.z + height_ / 2;

            obj_info.bounding_box_.dimensions.x = ((length_ < 0) ? -1 * length_ : length_);
            obj_info.bounding_box_.dimensions.y = ((width_ < 0) ? -1 * width_ : width_);
            obj_info.bounding_box_.dimensions.z = ((height_ < 0) ? -1 * height_ : height_);

            obj_list.push_back(obj_info);

        }


    }     */
    
 
    //  生成一个Tree对象
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    // 将点云数据按照Tree的方式进行存储，方便后续遍历
    tree -> setInputCloud(no_floor_cloud);
    std::vector<pcl::PointIndices> clusterIndices;
    // 生成欧氏距离聚类对象；
    pcl::EuclideanClusterExtraction<PointT> ec;
    // 设置欧式距离的buffer
    ec.setClusterTolerance(1);
    // 设置一个类别的最小数据点数
    ec.setMinClusterSize(MIN_CLUSTER_SIZE);
    //  设置一个类别的最大数据点数
    ec.setMaxClusterSize(MAX_CLUSTER_SIZE);
    // 搜索方法，Tree
    ec.setSearchMethod(tree);
    //  要搜索的输入
    ec.setInputCloud(no_floor_cloud);
    ec.extract(clusterIndices);

    vector<Detected_Obj> obj_list;
    // 遍历所有的聚类 
    for (size_t i = 0; i < clusterIndices.size(); i++)
    {
        // the structure to save one detected object
        Detected_Obj obj_info;

        float min_x = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_z = -std::numeric_limits<float>::max();
        // 遍历该聚类的所有点    采用索引方式  
        for (auto pit = clusterIndices[i].indices.begin(); pit != clusterIndices[i].indices.end(); ++pit)
        {
            //fill new colored cluster point by point
            pcl::PointXYZ p;
            p.x = no_floor_cloud->points[*pit].x;
            p.y = no_floor_cloud->points[*pit].y;
            p.z = no_floor_cloud->points[*pit].z;
            // 累计每个点的坐标  稍后需要求均值获取中心点  
            obj_info.centroid_.x += p.x;
            obj_info.centroid_.y += p.y;
            obj_info.centroid_.z += p.z;
            // 更新该聚类的范围  
            if (p.x < min_x)
                min_x = p.x;
            if (p.y < min_y)
                min_y = p.y;
            if (p.z < min_z)
                min_z = p.z;
            if (p.x > max_x)
                max_x = p.x;
            if (p.y > max_y)
                max_y = p.y;
            if (p.z > max_z)
                max_z = p.z;
        }

        // 更新聚类范围  用于绘制bounding box   
        obj_info.min_point_.x = min_x;
        obj_info.min_point_.y = min_y;
        obj_info.min_point_.z = min_z;

        obj_info.max_point_.x = max_x;
        obj_info.max_point_.y = max_y;
        obj_info.max_point_.z = max_z;

        // 计算聚类中心   

        obj_info.centroid_.x /= clusterIndices[i].indices.size();
        obj_info.centroid_.y /= clusterIndices[i].indices.size();
        obj_info.centroid_.z /= clusterIndices[i].indices.size();
        
        //calculate bounding box   box的size   
        double length_ = obj_info.max_point_.x - obj_info.min_point_.x;
        double width_ = obj_info.max_point_.y - obj_info.min_point_.y;
        double height_ = obj_info.max_point_.z - obj_info.min_point_.z;
        // 人和车 
        if(length_>=0.7||width_>=0.7||height_>=2||height_<=0.5)       // 一定不是人 
        {
            // 判断是否是车
            if(length_<=1||length_>=5||width_>=5||width_<=1||height_>=3||height_<=1)   continue;
        }


        obj_info.bounding_box_.header = cloud_header_;

        obj_info.bounding_box_.pose.position.x = obj_info.min_point_.x + length_ / 2;
        obj_info.bounding_box_.pose.position.y = obj_info.min_point_.y + width_ / 2;
        obj_info.bounding_box_.pose.position.z = obj_info.min_point_.z + height_ / 2;

        obj_info.bounding_box_.dimensions.x = length_ ;
        obj_info.bounding_box_.dimensions.y = width_;
        obj_info.bounding_box_.dimensions.z = height_ ;

        obj_list.push_back(obj_info);

    }
    
    // 发布框
    jsk_recognition_msgs::BoundingBoxArray bbox_array;

    for (size_t i = 0; i < obj_list.size(); i++)
    {
        bbox_array.boxes.push_back(obj_list[i].bounding_box_);
    }
    bbox_array.header = cloud_header_;

    pub_bounding_boxs.publish(bbox_array);
 
}


// 将激光投影到图像   
// 求得range图像 rangeMat
// 将点云按照模型重新设置索引为 fullCloud 
void projectPointCloud(pcl::PointCloud<PointT>::Ptr& laserCloudIn){
    float verticalAngle, horizonAngle, range;
    size_t rowIdn, columnIdn, index, cloudSize; 
    PointT thisPoint;

    cloudSize = laserCloudIn->points.size();
    // 遍历全部的点  
    for (size_t i = 0; i < cloudSize; ++i){
        // 获取XYZ的坐标   
        thisPoint.x = laserCloudIn->points[i].x;
        thisPoint.y = laserCloudIn->points[i].y;
        thisPoint.z = laserCloudIn->points[i].z;

        /*********** 计算 scanID 即垂直激光帧的序号 (根据lidar文档垂直角计算公式),根据仰角排列激光线号，velodyne每两个scan之间间隔2度）**************/
        // 计算点的仰角
        float angle = atan(thisPoint.z / sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
        if (N_SCANS == 16)
        {   // + 0.5 是用于四舍五入   因为 int 只会保留整数部分  如 int(3.7) = 3  
            rowIdn = int((angle + 15) / 2 + 0.5);    // +0.5是为了四舍五入, /2是每两个scan之间的间隔为2度，+15是过滤垂直上为[-,15,15]范围内

            if (rowIdn > (N_SCANS - 1) || rowIdn < 0)
            {   // 说明该点所处于的位置有问题  舍弃
                continue;
            }
        }                                // 下面两种为 32线与64线的   用的少
        else if (N_SCANS == 32)
        {
            rowIdn = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (rowIdn > (N_SCANS - 1) || rowIdn < 0)
            {
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                rowIdn = int((2 - angle) * 3.0 + 0.5);
            else
                rowIdn = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || rowIdn > 50 || rowIdn < 0)
            {
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }

        // atan2(y,x)函数的返回值范围(-PI,PI],表示与复数x+yi的幅角
        // 下方角度atan2(..)交换了x和y的位置，计算的是与y轴正方向的夹角大小(关于y=x做对称变换)
        // 这里是在雷达坐标系，所以是与正前方的夹角大小
        horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

        // round函数进行四舍五入取整
        // 这边确定不是减去180度???  不是
        // 雷达水平方向上某个角度和水平第几线的关联关系???关系如下：
        // horizonAngle:(-PI,PI],columnIdn:[H/4,5H/4]-->[0,H] (H:Horizon_SCAN)
        // 下面是把坐标系绕z轴旋转,对columnIdn进行线性变换
        // x+==>Horizon_SCAN/2,x-==>Horizon_SCAN
        // y+==>Horizon_SCAN*3/4,y-==>Horizon_SCAN*5/4,Horizon_SCAN/4
        //
        //          3/4*H
        //          | y+
        //          |
        // (x-)H---------->H/2 (x+)
        //          |
        //          | y-
        //    5/4*H   H/4
        //
        columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
        if (columnIdn >= Horizon_SCAN)
            columnIdn -= Horizon_SCAN;
        // 经过上面columnIdn -= Horizon_SCAN的变换后的columnIdn分布：
        //          3/4*H
        //          | y+
        //     H    |
        // (x-)---------->H/2 (x+)
        //     0    |
        //          | y-
        //         H/4
        //
        if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
            continue;
        // range图像   range为距离即像素值         
        range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
        rangeMat.at<float>(rowIdn, columnIdn) = range;    

        // columnIdn:[0,H] (H:Horizon_SCAN)==>[0,1800]   intensity 其实保存的是 该点的图像坐标   
        thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

        index = columnIdn  + rowIdn * Horizon_SCAN;
        fullCloud->points[index] = thisPoint;

    }
}

#if N_SCANS==16
const int groundScanInd = 8;
#elif N_SCANS==32
const int groundScanInd = 20;
#elif N_SCANS==64
const int groundScanInd = 40;
#endif
// 地面滤除
// 地面滤除    
void groundRemoval(){
    pcl::PointCloud<PointT>::Ptr floor_cloud(new pcl::PointCloud<PointT>());     //  地面的点云 
    vector<pair<int,int>> index;
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;
    for (size_t j = 0; j < Horizon_SCAN; ++j){
        // groundScanInd 是在 utility.h 文件中声明的线数，groundScanInd=7    地面点可能处于的位置  
        for (size_t i = 0; i < groundScanInd; ++i){
            // 上下两个点的序号   
            lowerInd = j + ( i )*Horizon_SCAN;
            upperInd = j + (i+1)*Horizon_SCAN;

            // 初始化的时候用nanPoint.intensity = -1 填充
            // 都是-1 证明不是激光点   
            if (fullCloud->points[lowerInd].intensity == -1 ||
                fullCloud->points[upperInd].intensity == -1){
                continue;
            }

            // 由上下两线之间点的XYZ位置得到两线之间的俯仰角
            // 如果俯仰角在10度以内，则判定(i,j)为地面点,groundMat[i][j]=1
            // 否则，则不是地面点，进行后续操作
            diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
            diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
            diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;
            // 计算倾斜角   
            angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;
            // 如果倾斜角小于10   则认为是地面   labelMat 赋值为-1   初始化为0      不参与聚类        
            if (abs(angle - sensorMountAngle) <= 30){
                /*
                floor_cloud->push_back(fullCloud->at(lowerInd));
                floor_cloud->push_back(fullCloud->at(upperInd));
                index.push_back(make_pair(i,j));
                index.push_back(make_pair(i+1,j)); 
                */
               labelMat.at<int>(i,j) = -1;
               labelMat.at<int>(i+1,j) = -1;
            }

        }
    }
    /*
    // 通过RANSAC进行筛选  
    // RANSAC 拟合平面 
    pcl::SampleConsensusModelPlane<PointT>::Ptr model_p(new pcl::SampleConsensusModelPlane<PointT>(floor_cloud));
    pcl::RandomSampleConsensus<PointT> ransac(model_p);
    ransac.setDistanceThreshold(1);       // 与该平面距离小于该阈值的点为内点 
    ransac.computeModel();
    // 获取内点  
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    ransac.getInliers(inliers->indices);

    // too few inliers   无地面点   
    if(inliers->indices.size() < floor_pts_thresh) {
      return;
    }
    else{    // 地面点提取成功   
      for(int i:inliers->indices)
        labelMat.at<int>(index[i].first,index[i].second) = -1;
    }  */
    //ROS_INFO_STREAM("floor points num: "<<;
}


// BFS 聚类 
// 如果聚类成功  则返回bounding box  
bool labelComponents(int row, int col, Detected_Obj& obj_info){
    auto t1 = ros::WallTime::now();
    float d1, d2,  angle;
    int fromIndX, fromIndY, thisIndX, thisIndY; 
    // 记录竖直方向是否有聚类   
    bool lineCountFlag[N_SCANS] = {false};

    // 数据队列    先push进第一个元素   
    queueIndX[0] = row;
    queueIndY[0] = col;
    // 队列大小   
    int queueSize = 1;
    // 队列头序号   
    int queueStartInd = 0;
    // 队列尾序号   
    int queueEndInd = 1;
    // 以放置的元素   用于对无效聚类重新设置label     
    allPushedIndX[0] = row;
    allPushedIndY[0] = col;
    int allPushedIndSize = 1;
    // 计算三角函数 
    double sin_X = sin(segmentAlphaX);
    double cos_X = cos(segmentAlphaX);
    double sin_Y = sin(segmentAlphaY);
    double cos_Y = cos(segmentAlphaY);
  //  ROS_INFO_STREAM("--------start cluster! !");
    // 标准的BFS
    // BFS的作用是以(row，col)为中心向外面扩散，
    // 判断(row,col)是否是这个平面中一点
    while(queueSize > 0){
        // 先取出队头的元素     queue.top()   
        fromIndX = queueIndX[queueStartInd];
        fromIndY = queueIndY[queueStartInd];
        --queueSize;                          // 队列内的元素数量--
        ++queueStartInd;
        // labelCount为全局遍历    初始值为1，后面会递增
        labelMat.at<int>(fromIndX, fromIndY) = labelCount;

        // neighbor=[[-1,0];[0,1];[0,-1];[1,0]]
        // 遍历点[fromIndX,fromIndY]边上的四个邻点
        for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
            // 邻点的坐标   
            thisIndX = fromIndX + (*iter).first;
            thisIndY = fromIndY + (*iter).second;

            if (thisIndX < 0 || thisIndX >= N_SCANS)
                continue;

            // 是个环状的图片，左右连通
            if (thisIndY < 0)
                thisIndY = Horizon_SCAN - 1;
            if (thisIndY >= Horizon_SCAN)
                thisIndY = 0;

            // 如果点[thisIndX,thisIndY]已经标记过
            // labelMat中，-1代表无效点，0代表未进行标记过，其余为其他的标记
            // 如果当前的邻点已经标记过，则跳过该点。
            // 如果labelMat已经标记为正整数，则已经聚类完成，不需要再次对该点聚类
            if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                continue;
            // 当前点以及其邻点   找到长的以及短的  
            d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                            rangeMat.at<float>(thisIndX, thisIndY));
            d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                            rangeMat.at<float>(thisIndX, thisIndY));

            // alpha代表角度分辨率，
            // X方向上角度分辨率是segmentAlphaX(rad)
            // Y方向上角度分辨率是segmentAlphaY(rad)
            if ((*iter).first == 0)
                angle = atan2(d2*sin_X, (d1 -d2*cos_X));     // x=0 y=+-1    表示左右 两根线    分辨率为 0.2
            else
                angle = atan2(d2*sin_Y, (d1 -d2*cos_Y));     // 上下两根线    

            // 通过下面的公式计算这两点之间是否有平面特征
            // atan2(y,x)的值越大，d1，d2之间的差距越小,越平坦
            
            // 若角度大于阈值  则归为一类  
            if (angle > segmentTheta){   // 若小于 segmentTheta 就不是一类了   
                // segmentTheta=1.0472<==>60度
                // 如果算出角度大于60度，则假设这是个平面
                // 放置到队尾  queue.push()      
                queueIndX[queueEndInd] = thisIndX;
                queueIndY[queueEndInd] = thisIndY;
                ++queueSize;
                ++queueEndInd;
                // 同一聚类的点用一个labelCount表示  
                labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                lineCountFlag[thisIndX] = true;        // 该竖直方向有聚类   
                // 记录被聚类的点  
                allPushedIndX[allPushedIndSize] = thisIndX;
                allPushedIndY[allPushedIndSize] = thisIndY;
                ++allPushedIndSize;      // 聚类的个数   
                if(allPushedIndSize>500)  return false;     // 聚类的点数太多直接退出
            }
        }
    }
    auto t2 = ros::WallTime::now();
    //cout<<" labelComponents time: " << (t2 - t1) * 1000.0 << " [msec]"<<endl;

    // 下面判定聚类是否有效    1. 聚类数大于30   2.垂直方向大于3  
    
    bool feasibleSegment = false;
  //  ROS_INFO_STREAM("cluster size: "<<allPushedIndSize);

    // 如果聚类超过30个点，直接标记为一个可用聚类，labelCount需要递增
    if (allPushedIndSize >= 30)
        feasibleSegment = true;
    else if (allPushedIndSize >= segmentValidPointNum){
        // 如果聚类点数小于30大于等于5，统计竖直方向上的聚类点数
        int lineCount = 0;
        for (size_t i = 0; i < N_SCANS; ++i)
            if (lineCountFlag[i] == true)
                ++lineCount;

        // 竖直方向上超过3个也将它标记为有效聚类
        if (lineCount >= segmentValidLineNum)
            feasibleSegment = true;            
    }
    
    
    // 该聚类有效则 labelCount++  
    if (feasibleSegment == true){
        ++labelCount;
        
        float min_x = std::numeric_limits<float>::max();
        float max_x = -std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_y = -std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_z = -std::numeric_limits<float>::max();
        // 求解bounding box 
        for (size_t i = 0; i < allPushedIndSize; ++i){
                //fill new colored cluster point by point
                pcl::PointXYZ p;
                p.x = fullCloud->points[allPushedIndX[i]*Horizon_SCAN+allPushedIndY[i]].x;
                p.y = fullCloud->points[allPushedIndX[i]*Horizon_SCAN+allPushedIndY[i]].y;
                p.z = fullCloud->points[allPushedIndX[i]*Horizon_SCAN+allPushedIndY[i]].z;
                // 累计每个点的坐标  稍后需要求均值获取中心点  
                obj_info.centroid_.x += p.x;
                obj_info.centroid_.y += p.y;
                obj_info.centroid_.z += p.z;
                // 更新该聚类的范围  
                if (p.x < min_x)
                    min_x = p.x;
                if (p.y < min_y)
                    min_y = p.y;
                if (p.z < min_z)
                    min_z = p.z;
                if (p.x > max_x)
                    max_x = p.x;
                if (p.y > max_y)
                    max_y = p.y;
                if (p.z > max_z)
                    max_z = p.z;
        
        }
        // 更新聚类范围  用于绘制bounding box   
        obj_info.min_point_.x = min_x;
        obj_info.min_point_.y = min_y;
        obj_info.min_point_.z = min_z;

        obj_info.max_point_.x = max_x;
        obj_info.max_point_.y = max_y;
        obj_info.max_point_.z = max_z;

        // 计算聚类中心   

        obj_info.centroid_.x /= allPushedIndSize;
        obj_info.centroid_.y /= allPushedIndSize;
        obj_info.centroid_.z /= allPushedIndSize;
        
        //calculate bounding box   box的size   
        double length_ = obj_info.max_point_.x - obj_info.min_point_.x;
        double width_ = obj_info.max_point_.y - obj_info.min_point_.y;
        double height_ = obj_info.max_point_.z - obj_info.min_point_.z;

        //if(length_>=5||width_>=5||height_>=5)   continue;

        obj_info.bounding_box_.pose.position.x = obj_info.min_point_.x + length_ / 2;
        obj_info.bounding_box_.pose.position.y = obj_info.min_point_.y + width_ / 2;
        obj_info.bounding_box_.pose.position.z = obj_info.min_point_.z + height_ / 2;

        obj_info.bounding_box_.dimensions.x = length_ ;
        obj_info.bounding_box_.dimensions.y = width_;
        obj_info.bounding_box_.dimensions.z = height_ ;

        // 通过框的size进一步筛选
        if((length_>3||length_<6)&&(width_>3||width_<6)&&(height_>1.5||height_<3)) 
        {
           
           return true; 
        }
    }else{
        // 遍历所有放置的点  将label设置为999999 表示错误   
        for (size_t i = 0; i < allPushedIndSize; ++i){
            // 标记为999999的是需要舍弃的聚类的点，因为他们的数量小于30个
            labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
        }
    }
    return false;   
}

// 基于角度与距离约束的聚类  
void cloudSegmentation(std_msgs::Header cloud_header_, pcl::PointCloud<PointT>::Ptr& laserCloudIn){
    vector<Detected_Obj> obj_list;
    Detected_Obj box;
    auto t1 = ros::WallTime::now();
    // 对点进行聚类 
    for (size_t i = 0; i < N_SCANS; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            // 如果labelMat[i][j]=0,表示没有对该点进行过分类
            // 需要对该点进行聚类  
            if (labelMat.at<int>(i,j) == 0){
            //    ROS_INFO_STREAM("null floor!! ");
                // 进行聚类并返回bounding box   
                if(labelComponents(i, j, box)){
                  box.bounding_box_.header = cloud_header_;
                  obj_list.push_back(box);
                }    
                  
            }
        }
    }
    auto t2 = ros::WallTime::now();
    cout<<" cluster time: " << (t2 - t1) * 1000.0 << " [msec]"<<endl;

    ROS_INFO_STREAM("segment num: "<<obj_list.size());

    ROS_INFO_STREAM("raw points size: "<<laserCloudIn->size());
    laserCloudIn->clear();

    // 聚类完成后   labelMat 的分布是 : -1 地面点    999999: 离群点    1-999+   有效聚类点    
    for (size_t i = 0; i < N_SCANS; ++i) {
        for (size_t j = 0; j < Horizon_SCAN; ++j) {
            // 如果为地面点  或其他分割的点  
            if (labelMat.at<int>(i,j) > 0) {
               laserCloudIn->push_back(fullCloud->at(i*Horizon_SCAN+j));  
        
            }
        }
    }

    ROS_INFO_STREAM("after filter points size: "<<laserCloudIn->size());
    // 发布框
    jsk_recognition_msgs::BoundingBoxArray bbox_array;

    for (size_t i = 0; i < obj_list.size(); i++)
    {
        bbox_array.boxes.push_back(obj_list[i].bounding_box_);
    }
    bbox_array.header = cloud_header_;

    pub_bounding_boxs.publish(bbox_array);

}

PointT nanPoint;

// 初始化/重置各类参数内容
void resetParameters(){

    rangeMat = cv::Mat(N_SCANS, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    labelMat = cv::Mat(N_SCANS, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
    labelCount = 1;

    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
}

void laser_callback(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
    // ros 转 PCL 
    const auto& stamp = laserCloudMsg->header.stamp;
    pcl::PointCloud<PointT>::Ptr laserCloudIn(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);

    std::vector<int> idx;
    pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, idx);
    if(laserCloudIn->empty()) {
      ROS_INFO("cloud is empty!!");
      return;
    }
    auto t1 = ros::WallTime::now();
    #if use_fast_segmentation
    projectPointCloud(laserCloudIn);
    groundRemoval();
    cloudSegmentation(laserCloudMsg->header,laserCloudIn);
    laserCloudIn = downsample(laserCloudIn);          // 降采样滤波  
    resetParameters();
    #else    
    laserCloudIn = distance_filter(laserCloudIn);     // 距离滤波 
    laserCloudIn = downsample(laserCloudIn);          // 降采样滤波  
    laserCloudIn = outlier_removal(laserCloudIn);     // 离群点去除   

    // 如果开启地面去除的话
    if(floor_remove_enable){
        laserCloudIn = floor_remove(laserCloudIn); 
     //   cluster_segment(laserCloudIn, 5, 
     //           100, laserCloudMsg->header);
    }
    laserCloudIn = outlier_removal(laserCloudIn);     // 离群点去除
    #endif
    auto t3 = ros::WallTime::now();
    cout<<" filter time: " << (t3 - t1) * 1000.0 << " [msec]"<<endl;
    // 发布
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*laserCloudIn, output);
    output.header.stamp=ros::Time::now(); 
    output.header.frame_id = laserCloudMsg->header.frame_id;
    // cout<<"frame_id: "<<output.header.frame_id<<endl;
    points_pub.publish(output);
    
}


void init(ros::NodeHandle &n)
{
   float downsample_resolution = n.param<double>("downsample_resolution", 0.1); 
   cout<<"downsample: VOXELGRID,resolution: "<<downsample_resolution<<endl;
   // 创建指向体素滤波器VoxelGrid的智能指针    
   boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
   voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
   downsample_filter = voxelgrid;
   /***************** 离群点滤波器初始化 ***************************/
    double radius = n.param<double>("radius_r", 0.5);                  
    int min_neighbors = n.param<int>("radius_min_neighbors", 2);
    std::cout << "outlier_removal: RADIUS " << radius << " - " << min_neighbors << std::endl;
    pcl::RadiusOutlierRemoval<PointT>::Ptr rad(new pcl::RadiusOutlierRemoval<PointT>());     // 几何法去除离群点  
    rad->setRadiusSearch(radius);                      
    rad->setMinNeighborsInRadius(min_neighbors);
    outlier_removal_filter = rad;
    // 距离滤波初始化
    distance_far_thresh = n.param<double>("distance_far_thresh", 100);
    distance_near_thresh = n.param<double>("distance_near_thresh",  0.1);
    cout<<"distance filter threshold: "<< distance_near_thresh << " ,"<<distance_far_thresh<<endl;

    normal_filter_thresh = n.param<double>("normal_filter_thresh", 45.0);
    clip_height = n.param<double>("clip_height", -0.0);
    
    rangeMat = cv::Mat(N_SCANS, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    labelMat = cv::Mat(N_SCANS, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
    fullCloud.reset(new pcl::PointCloud<PointT>());
    fullCloud->points.resize(N_SCANS*Horizon_SCAN);

    // labelComponents函数中用到了这个矩阵
    // 该矩阵用于求某个点的上下左右4个邻接点
    std::pair<int8_t, int8_t> neighbor;
    neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
    neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);
}

int main(int argc, char **argv)
{
    ros::init (argc, argv, "preprocess_node");   
    ROS_INFO("Started preprocess_node");   
    ros::NodeHandle nh("~");                         // 初始化私有句柄   
    init(nh);
    // 雷达订阅   
    laser_sub = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 5, laser_callback);       
    points_pub =  nh.advertise<sensor_msgs::PointCloud2> ("/processed_points", 5);                   // 初始化发布器    
    floor_pub =  nh.advertise<sensor_msgs::PointCloud2> ("/floor_points", 5);            
    pub_bounding_boxs = nh.advertise<jsk_recognition_msgs::BoundingBoxArray> ("/detected_bounding_boxs", 5);       // 初始化发布器      
    ros::spin(); 
    return 0;
}