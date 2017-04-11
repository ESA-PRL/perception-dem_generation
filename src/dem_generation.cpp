#include "dem_generation.hpp"
#include <iostream>

using namespace std;
using namespace dem_generation;


// creator
DEM::DEM() 
   // : gaussian_kernel(0), calibrationInitialized( false )
{
	
	camera_set = 0;
	timestamp_set = 0;

	cloud_input_p.reset( new pcl::PointCloud<pcl::PointXYZ> );
	cloud_filtered_p.reset( new pcl::PointCloud<pcl::PointXYZ> );
	
}

void DEM::welcome()
{
    cout << "You successfully compiled and executed DummyProject. Welcome!" << endl;
}

void DEM::setCameraParameters(int width, int height, float cx, float cy, float fx, float fy)
{
	this->width = width;
	this->height = height;
	this->cx = cx;
	this->cy = cy;
	this->fx = fx;
	this->fy = fy;
	
	// reserve space
	cloud_input_p->width    = width;
	cloud_input_p->height   = height;
	cloud_input_p->is_dense = false;
	cloud_input_p->points.resize (width*height);
	
	xArray_const.reserve(height*width); 
	yArray_const.reserve(height*width); 
	xArray.reserve(height*width); 
  	yArray.reserve(height*width); 
  	
  	int ix = 1;
  	for(int i = 1; i<=height; i++)
  	{
		for(int j = 1; j<=width; j++)
		{
			yArray_const.push_back(ix);
			xArray_const.push_back(j);
		}
		ix++;
	}
	
	// prepare x and y arrays
	transform(xArray_const.begin(), xArray_const.end(), xArray_const.begin(),
          bind1st(std::plus<float>(), -cx)); 
    transform(yArray_const.begin(), yArray_const.end(), yArray_const.begin(),
          bind1st(std::plus<float>(), -cy)); 
            
    transform(xArray_const.begin(), xArray_const.end(), xArray_const.begin(),
          bind1st(std::multiplies<float>(), 1/fx)); 
    transform(yArray_const.begin(), yArray_const.end(), yArray_const.begin(),
          bind1st(std::multiplies<float>(), 1/fy));             
          
	camera_set = 1;
}

void DEM::setTimestamp(std::string timestamp)
{
	// save timestamp
	timestamp.erase(std::remove(timestamp.begin(),timestamp.end(), ':' ), timestamp.end() ) ;
	sensor_data_time = timestamp;
	
	timestamp_set = 1;
}

void DEM::setColorFrame(cv::Mat color_frame_left, cv::Mat color_frame_right)
{	
	if(!timestamp_set)
		std::cerr << "The timestamp has never been set!\n";  	
	
	// receive frame from orogen, save it in a document, keep the path name and save it to internal variable
	color_frame_location_left = default_save_location + camera_name + "_" + sensor_data_time + "_left.png";
	cv::imwrite(color_frame_location_left, color_frame_left);
	
	if ((camera_name=="FLOC_STEREO") || (camera_name=="RLOC_STEREO"))
	{
		color_frame_location_right = default_save_location + camera_name + "_" + sensor_data_time + "_right.png";
		cv::imwrite(color_frame_location_right, color_frame_right);
	}
}

void DEM::setFileDestination(std::string default_save_location, std::string camera_name)
{
	this->default_save_location = default_save_location;
	this->camera_name = camera_name;
}

void DEM::distance2pointCloud(std::vector<float> distance)
{
	if(!camera_set)
		std::cerr << "The camera properties have not been set yet!\n";  	
	
	// reserve space tbd is this needed here??
	cloud_input_p->width    = width;
	cloud_input_p->height   = height;
	cloud_input_p->is_dense = false;
	cloud_input_p->points.resize (width*height);
	
	std::transform( xArray_const.begin(), xArray_const.end(),
					distance.begin(), xArray.begin(), 
					std::multiplies<float>() );
			
	std::transform( yArray_const.begin(), yArray_const.end(),
					distance.begin(), yArray.begin(), 
					std::multiplies<float>() );
					
	for(unsigned int i = 0; i<cloud_input_p->size(); i++)
	{
		cloud_input_p->points[i].x = xArray[i];
		cloud_input_p->points[i].y = yArray[i];
		cloud_input_p->points[i].z = distance[i];
	}

	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud_input_p,*cloud_input_p, indices);
	
}

void DEM::pointCloud2Mesh()
{
	if(!timestamp_set)
		std::cerr << "The timestamp has never been set!\n";  
		
	Eigen::Vector4f	filter_box_min, filter_box_max;
	
	filter_box_min[0] = -4.0; 
	filter_box_min[1] = -4.0; 
	filter_box_min[2] = 0.5; 
	
	filter_box_max[0] = 4.0; 
	filter_box_max[1] = 4.0; 
	filter_box_max[2] = 4.0; 
	
	// Voxel grid
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud (cloud_input_p);					
	sor.setLeafSize (0.05f, 0.05f, 0.05f);
	sor.filter (*cloud_filtered_p);				
	
	// remove out of bound points
	// ATTENTION, if this is performed before the voxel filter,
	// then there will be degenerate faces in the mesh (dont know why)
	// but the first voxel grid may fail because there are too
	// many points, so it has to be executed again after the crop box
	pcl::CropBox<pcl::PointXYZ> cb;
	cb.setInputCloud(cloud_filtered_p);
	cb.setMin(filter_box_min);
	cb.setMax(filter_box_max);
	cb.filter(*cloud_filtered_p);
	
	sor.setInputCloud (cloud_filtered_p);					
	sor.setLeafSize (0.05f, 0.05f, 0.05f);
	sor.filter (*cloud_filtered_p);	
	
	// Normal estimation*
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud (cloud_filtered_p);
	n.setInputCloud (cloud_filtered_p);
	n.setSearchMethod (tree);
	n.setKSearch(20);
	n.compute (*normals);
	
	//* normals should not contain the point normals + surface curvatures
	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields (*cloud_filtered_p, *normals, *cloud_with_normals);
	//* cloud_with_normals = cloud + normals
	//Flip all normals towards the viewer
	for(unsigned int i=0; i<cloud_with_normals->width*cloud_with_normals->height; i++){
		pcl::flipNormalTowardsViewpoint(cloud_with_normals->points[i], 0.0f, 0.0f, 0.0f, cloud_with_normals->points[i].normal_x, cloud_with_normals->points[i].normal_y, cloud_with_normals->points[i].normal_z);
	}
	// Create search tree*
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud (cloud_with_normals);
	// Initialize objects
	pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
	pcl::PolygonMesh triangles;

	// Set the maximum distance between connected points (maximum edge length)
	gp3.setSearchRadius (0.09);
	// Set typical values for the parameters
	gp3.setMu (2.5);
	gp3.setMaximumNearestNeighbors (100);
	gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
	gp3.setMinimumAngle(M_PI/18); // 10 degrees
	gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
	gp3.setNormalConsistency(true);
	gp3.setConsistentVertexOrdering(true);
	// Get result
	gp3.setInputCloud (cloud_with_normals);
	gp3.setSearchMethod (tree2);
	gp3.reconstruct (triangles);

	// Additional vertex information
	std::vector<int> parts = gp3.getPartIDs();
	std::vector<int> states = gp3.getPointStates();

	//Texture the mesh

	// test texture mapping
	pcl::TextureMapping<pcl::PointXYZ> tm;
	// read current camera

    std::vector<std::string> color_frame_location_vect; 
    color_frame_location_vect.push_back(color_frame_location_left);  // TODO CLEAN UP

	// IS ALL THIS USED??
	pcl::texture_mapping::CameraVector cam_vector;
	pcl::texture_mapping::Camera cam;
	cam.pose = Eigen::Affine3f::Identity(); 
	cam.focal_length = fx;
	cam.height = height;
	cam.width = width;
	cam.texture_file = color_frame_location_left;				
	cam_vector.push_back(cam);

    pcl::TexMaterial tex_material;

    // default texture materials parameters
    tex_material.tex_Ka.r = 0.2f;
    tex_material.tex_Ka.g = 0.2f;
    tex_material.tex_Ka.b = 0.2f;

    tex_material.tex_Kd.r = 0.8f;
    tex_material.tex_Kd.g = 0.8f;
    tex_material.tex_Kd.b = 0.8f;

    tex_material.tex_Ks.r = 1.0f;
    tex_material.tex_Ks.g = 1.0f;
    tex_material.tex_Ks.b = 1.0f;
    tex_material.tex_d = 1.0f;
    tex_material.tex_Ns = 0.0f;
    tex_material.tex_illum = 2;

    // set texture material paramaters
    tm.setTextureMaterials(tex_material);

    // set 2 texture for 2 mesh
    std::vector<std::string> tex_files;
    tex_files = color_frame_location_vect;

    // set texture files
    tm.setTextureFiles(tex_files);

    // initialize texture mesh
    pcl::TextureMesh tex_mesh;
    tex_mesh.header = triangles.header;
    tex_mesh.cloud = triangles.cloud;
           
    std::vector<pcl::Vertices> polygon1;

        for(size_t i =0; i < triangles.polygons.size(); ++i){
                polygon1.push_back(triangles.polygons[i]);      
        }

    tex_mesh.tex_polygons.push_back(polygon1);
	
	// mapping
    mapTexture2MeshUVnew(tex_mesh, tex_material, tex_files);
    
    // Save obj, view in meshlab.	
    mesh_location = default_save_location + camera_name + "_" + sensor_data_time + ".obj";
    pcl::io::saveOBJFile (mesh_location, tex_mesh , 6); 
}


void DEM::mapTexture2MeshUVnew (pcl::TextureMesh &tex_mesh, pcl::TexMaterial &tex_material, std::vector<std::string> &tex_files)
{
	// mesh information
	int nr_points = tex_mesh.cloud.width * tex_mesh.cloud.height;
	int point_size = static_cast<int> (tex_mesh.cloud.data.size ()) / nr_points;

	float x_lowest = 100000;
	float x_highest = 0;
	float y_lowest = 100000;
	//float y_highest = 0 ;
	float z_lowest = 100000;
	float z_highest = 0;
	float x_, y_, z_;
	for (int i = 0; i < nr_points; ++i)
	{
		memcpy (&x_, &tex_mesh.cloud.data[i * point_size + tex_mesh.cloud.fields[0].offset], sizeof(float));
		memcpy (&y_, &tex_mesh.cloud.data[i * point_size + tex_mesh.cloud.fields[1].offset], sizeof(float));
		memcpy (&z_, &tex_mesh.cloud.data[i * point_size + tex_mesh.cloud.fields[2].offset], sizeof(float));
		// x
		if (x_ <= x_lowest)
		x_lowest = x_;
		if (x_ > x_lowest)
		x_highest = x_;
		
		// y
		if (y_ <= y_lowest)
		y_lowest = y_;
		//if (y_ > y_lowest) y_highest = y_;
		
		// z
		if (z_ <= z_lowest)
		z_lowest = z_;
		if (z_ > z_lowest)
		z_highest = z_;
	}
	
	
	// texture coordinates for each mesh
	std::vector<std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > >texture_map;
	
	for (size_t m = 0; m < tex_mesh.tex_polygons.size (); ++m)
	{
		// texture coordinates for each mesh
		std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > texture_map_tmp;
		
		// processing for each face
		for (size_t i = 0; i < tex_mesh.tex_polygons[m].size (); ++i)
		{
			size_t idx;
			Eigen::Vector2f tmp_VT;
			for (size_t j = 0; j < tex_mesh.tex_polygons[m][i].vertices.size (); ++j)
			{
				idx = tex_mesh.tex_polygons[m][i].vertices[j];
				memcpy (&x_, &tex_mesh.cloud.data[idx * point_size + tex_mesh.cloud.fields[0].offset], sizeof(float));
				memcpy (&y_, &tex_mesh.cloud.data[idx * point_size + tex_mesh.cloud.fields[1].offset], sizeof(float));
				memcpy (&z_, &tex_mesh.cloud.data[idx * point_size + tex_mesh.cloud.fields[2].offset], sizeof(float));
				
				// calculate uv coordinates
				tmp_VT[0] = ((fx * x_ / z_) +cx - 1 ) /width;
				tmp_VT[1] = 1 - ((fy * y_ / z_) +cy - 1 ) /height;
				//cout << ((fx * x_ / z_) +cx - 1 ) << ", " << ((fy * y_ / z_) +cy - 1 ) << endl;
				texture_map_tmp.push_back (tmp_VT);
			}
		}// end faces
		
		// texture materials
		std::stringstream tex_name;
		tex_name << "material_" << m;
		tex_name >> tex_material.tex_name;
		tex_material.tex_file = tex_files[m];
		tex_mesh.tex_materials.push_back (tex_material);
		
		// texture coordinates
		tex_mesh.tex_coordinates.push_back (texture_map_tmp);
	}// end meshes
}

void DEM::saveDistanceFrame(std::vector<float> distance)
{
	if(!timestamp_set)
		std::cerr << "The timestamp has never been set!\n";  
		
	cv::Mat tmp = cv::Mat(distance).reshape(0,height);
	
	cv::Mat distance_frame(height, width, CV_8UC4, tmp.data);
	
	distance_frame_location = default_save_location + camera_name + "_" + sensor_data_time + "_distance.bmp";
	cv::imwrite(distance_frame_location, distance_frame);	
}

void DEM::savePointCloud()
{
	if(!timestamp_set)
		std::cerr << "The timestamp has never been set!\n";  
		
	point_cloud_obj_location = default_save_location + camera_name + "_" + sensor_data_time + "_pc.obj";

    pcl::PolygonMesh mesh;
    pcl::toPCLPointCloud2(*cloud_filtered_p, mesh.cloud);
    pcl::io::saveOBJFile(point_cloud_obj_location, mesh);

}


std::string DEM::getMeshPath()
{
	return mesh_location;
}


std::string DEM::getImageLeftPath()
{
	return color_frame_location_left;
}

std::string DEM::getImageRightPath()
{
	return color_frame_location_right;
}

std::string DEM::getDistanceFramePath()
{
	return distance_frame_location;
}

	/*  const int rows_p= height*width;
  const int cols_p= 3;
  double **points = (double **) malloc(sizeof(double *)*rows_p);
  for(int i=0; i<rows_p; i++){
	// Allocate array, store pointer  
	points[i] = (double *) malloc(sizeof(double)*cols_p); 
  }  

  cloud_input_p->width    = width;
  cloud_input_p->height   = height;
  cloud_input_p->is_dense = false;
  cloud_input_p->points.resize (width*height);

	// std::numeric_limits<float>::quiet_NaN()
	std::cout << "check2" << std::endl;

  
  int valid_points=0;

  int idxi=0; int idxj=-1; int idx=-1;
  const float bad_point = std::numeric_limits<float>::quiet_NaN();

 


  
  for(int opp = 0; opp < width*height; opp++)
  {
	idxj++; idx++;
	if (idxj>width-1){
		idxj=0;idxi++;
	}
	//line.erase (0,2);
//std::cout << "line: " << line << " idx: " << idx << std::endl;

	if (1==0)//line.compare("nan") == 0)
	{
		points[idx][0]=idxi;
		points[idx][1]=idxj;
		points[idx][2]=0;
		//std::cout << "no!!!!!!" << idx << std::endl;
		//valid_points++;
		//cloud->points[valid_points].z= bad_point;
		//cloud->points[valid_points].x= bad_point;
		//cloud->points[valid_points].y= bad_point;

	}else{
		
		//cout<< "ei" << endl;
		valid_points++;
        	points[idx][0]=idxi;
		points[idx][1]=idxj;
		points[idx][2]=distance[opp];


        	cloud_input_p->points[valid_points].z= points[idx][2];
		cloud_input_p->points[valid_points].x= cloud_input_p->points[valid_points].z * (points[idx][1]+ 1 - cx)/ fx;
		cloud_input_p->points[valid_points].y= cloud_input_p->points[valid_points].z * (points[idx][0]+ 1 - cy)/ fy;
		//float tmp = cloud->points[valid_points].z;
		//cloud->points[valid_points].x= cloud->points[valid_points].y;
		//cloud->points[valid_points].z = cloud->points[valid_points].y;
        	//cloud->points[valid_points].y = tmp; 
		

	}
	
  }*/
