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
            void welcome();
            void setCameraParameters(int width, int height, float cx, float cy, float fx, float fy);
            void setColorFrame(cv::Mat color_frame_left, cv::Mat color_frame_right);
            void setFileDestination(std::string default_save_location, std::string camera_name);
            void distance2pointCloud(std::vector<float> distance);
            void pointCloud2Mesh();
            void mapTexture2MeshUVnew(pcl::TextureMesh &tex_mesh, pcl::TexMaterial &tex_material, std::vector<std::string> &tex_files);
            void saveDistanceFrame(std::vector<float> distance);
			void savePointCloud();
			std::string getMeshPath();
            std::string getImageLeftPath();
            std::string getImageRightPath();
			std::string getDistanceFramePath();

            
		private:
		
			int camera_set;
                        int processing_count;

		
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
    };

} // end namespace dem_generation

#endif // _DUMMYPROJECT_DUMMY_HPP_
