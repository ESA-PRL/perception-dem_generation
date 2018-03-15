#ifndef _DEM_GENERATION_HPP_
#define _DEM_GENERATION_HPP_


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <iostream>
#include <fstream>
#include <string>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/file_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/texture_mapping.h>
#include <time.h>

//#include <pcl/surface/3rdparty/opennurbs/opennurbs_mesh.h>

#include <sys/time.h>


namespace dem_generation
{
    class DEM
    {
        public:

            DEM();
            void setCameraParameters(int width, int height, float cx, float cy, float fx, float fy);
            void setPcFiltersParameters(Eigen::Vector4f filter_box_min, Eigen::Vector4f filter_box_max, float leaf_size, int k_points);
            void setTimestamp(std::string sensor_data_time);
            void setColorFrame(cv::Mat color_frame);
            void setColorFrameStereo(cv::Mat color_frame_left, cv::Mat color_frame_right);
            void setFileDestination(std::string default_save_location, std::string camera_name);
            void compressProducts(bool enable_compression, int compression_level);
            void distance2pointCloud(std::vector<float> distance);
            void setPointCloud(pcl::PointCloud<pcl::PointXYZ>& input_cloud);
            void setPointCloud(std::vector<Eigen::Vector3d>& input_cloud);
            void filterPointCloud();
            int pointCloud2Mesh(bool use_filtered);
            void mapTexture2MeshUVnew(pcl::TextureMesh &tex_mesh, pcl::TexMaterial &tex_material, std::vector<std::string> &tex_files);
            void saveDistanceFrame(std::vector<float> distance);
            void savePointCloud(bool filtered);
            std::string getMeshPath();
            std::string getPointCloudPath();
            std::string getImageLeftPath();
            std::string getImageRightPath();
            std::string getDistanceFramePath();


        private:

            int camera_set;
            //int processing_count;
            int timestamp_set;
            int filter_set;
            int pc_set;
            int pc_filtered;
            bool compression_enabled;
            int compression_level;

            // camera parameters
            float width;
            float height;
            float cx;
            float cy;
            float fx;
            float fy;

            // arrays used to compute the pointcloud
            std::vector<float> xArray_const, yArray_const, xArray, yArray;

            // pointclouds
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input_p, cloud_filtered_p;

            // location of the saved color frame and camera name
            std::string color_frame_location_left, color_frame_location_right,
                    mesh_location, default_save_location, camera_name,
                    distance_frame_location, point_cloud_ply_location,
                    point_cloud_obj_location;

            // data capture time string
            std::string sensor_data_time;

            // Point cloud filter parameters
            Eigen::Vector4f filter_box_min, filter_box_max;
            float leaf_size;
            int k_points;
            std::string constructProductPath(std::string, std::string);

    };

} // end namespace dem_generation

#endif // _DUMMYPROJECT_DUMMY_HPP_
