#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <pcl/filters/voxel_grid.h>

using namespace std;

typedef pcl::PointXYZI PointT;

ros::Publisher map_pub;          // 匹配地图发布    
ros::Publisher globalmap_pub; 


pcl::PointCloud<PointT>::Ptr Matched_map;      // 保存地图  

// 初始地图发布    
void initializeMap_pub(ros::NodeHandle& n) {
// read globalmap from a pcd file
std::string globalmap_pcd = n.param<std::string>("globalmap_pcd", "");
cout<<"global map pcd: "<<globalmap_pcd<<endl;
Matched_map.reset(new pcl::PointCloud<PointT>());
pcl::io::loadPCDFile(globalmap_pcd, *Matched_map);               // 读取PCD到 globalmap 文件  
cout<<"map raw size: "<<Matched_map->size()<<endl;

Matched_map->header.frame_id = "map";                              // 设定地图坐标系  
// downsample globalmap    对地图执行降采样  
double downsample_resolution = n.param<double>("downsample_resolution", 0.1);
boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
voxelgrid->setInputCloud(Matched_map);
pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
voxelgrid->filter(*filtered);
Matched_map = filtered;
cout<<"map after filter size: "<<Matched_map->size()<<endl;
map_pub.publish(Matched_map);
}


int main(int argc, char **argv)
{
    ros::init (argc, argv, "MapServer_node");   
    ROS_INFO("Started MapServer_node");   
    ros::NodeHandle nh("~");                              // 初始化私有句柄   
 //   map_pub = nh.advertise<sensor_msgs::PointCloud2>("/map", 5, true);    // 地图发布话题    为什么设置成5  以及  true  
 //   pose_sub = nh.subscribe("/odom", 1, pose_callback);
    map_pub = nh.advertise<sensor_msgs::PointCloud2>("/map", 1);
    globalmap_pub = nh.advertise<sensor_msgs::PointCloud2>("/globalmap", 1);

    sleep(2);
    initializeMap_pub(nh);
    ros::Rate r(1);   
    int i=0;
    while (ros::ok())
    {           
    r.sleep();
    }
    return 0;
}