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
    filter_set = 0;
    pc_set = 0;
    pc_filtered = 0;
    compression_enabled = 0;

    cloud_input_p.reset( new pcl::PointCloud<pcl::PointXYZ> );
    cloud_filtered_p.reset( new pcl::PointCloud<pcl::PointXYZ> );
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

void DEM::setPcFiltersParameters(Eigen::Vector4f filter_box_min, Eigen::Vector4f filter_box_max, float leaf_size, int k_points)
{
    filter_set = 1;
    this->leaf_size = leaf_size;
    this->k_points = k_points;
    this->filter_box_min = filter_box_min;
    this->filter_box_max = filter_box_max;
}


void DEM::setTimestamp(std::string timestamp)
{
    // save timestamp
    timestamp.erase(std::remove(timestamp.begin(),timestamp.end(), ':' ), timestamp.end() ) ;
    sensor_data_time = timestamp;

    timestamp_set = 1;
}

void DEM::setColorFrame(const cv::Mat& color_frame)
{
    // receive frame from orogen, save it in a document, keep the path name and save it to internal variable
    if(compression_enabled)
    {
        vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
        compression_params.push_back(compression_level);
        color_frame_location_left = constructProductPath("IMAGE",".jpg","LEFT");
        cv::imwrite(color_frame_location_left, color_frame, compression_params);
    }
    else
    {
        color_frame_location_left = constructProductPath("IMAGE",".png","LEFT");
        cv::imwrite(color_frame_location_left, color_frame);
    }
}

std::string DEM::constructProductPath(std::string identifier, std::string file_ending, std::string left_right)
{
    // append left/right to sensor name if necessary
    // e.g. LOCCAM_LEFT_IMAGE but not in LOCCAM_DEM
    if (left_right == "LEFT" || left_right == "RIGHT")
    {
        left_right += "_";
    }
    else
    {
        left_right = "";
    }

    return default_save_location + camera_name + "_" + left_right + identifier + "_" + sensor_data_time + file_ending;
}

void DEM::setColorFrameStereo(const cv::Mat& color_frame_left, const cv::Mat& color_frame_right)
{
    // receive frame from orogen, save it in a document, keep the path name and save it to internal variable
    if(compression_enabled)
    {
        vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
        compression_params.push_back(compression_level);

        color_frame_location_left  = constructProductPath("IMAGE", ".jpg", "LEFT");
        color_frame_location_right = constructProductPath("IMAGE", ".jpg", "RIGHT");

        cv::imwrite(color_frame_location_left,  color_frame_left, compression_params);
        cv::imwrite(color_frame_location_right, color_frame_right, compression_params);
    }
    else
    {
        color_frame_location_left  = constructProductPath("IMAGE", ".png", "LEFT");
        color_frame_location_right = constructProductPath("IMAGE", ".png", "RIGHT");

        cv::imwrite(color_frame_location_left,  color_frame_left);
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
        LOG_WARN_S << "The camera properties have not been set yet!";

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

    pc_set = 1;
    pc_filtered = 0;
}

void DEM::setPointCloud(pcl::PointCloud<pcl::PointXYZ>& input_cloud)
{
    cloud_input_p->swap(input_cloud);
    cloud_input_p->resize(input_cloud.size()); // when saving unfiltered, handles the possibility the size of the pointclous is changing

    Eigen::Matrix3d m;
    // pointcloud of tof needs to be flipped 180 around Z axis. At the moment this is a custom solution
    Eigen::Quaterniond attitude;
    attitude = Eigen::Quaternion <double> (Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ())*
                            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX()));
    Eigen::Translation<double,3> ptu2Bd(Eigen::Vector3d(0.0,0.0,0.0));

    Eigen::Transform<double,3,Eigen::Affine> combined = attitude*ptu2Bd;


    pcl::transformPointCloud(*cloud_input_p, *cloud_input_p, combined);

    pc_set = 1;
    pc_filtered = 0;
}

void DEM::setPointCloud(std::vector<Eigen::Vector3d>& input_cloud)
{
    // TODO??
    pc_set = 1;
    pc_filtered = 0;
}

void DEM::compressProducts(bool enable_compression, int compression_level)
{
    compression_enabled = enable_compression;
    this->compression_level = compression_level;
    if(this->compression_level > 100)
        this->compression_level = 100;
    if(this->compression_level < 0)
        this->compression_level = 0;
}

void DEM::filterPointCloud()
{
    if(!filter_set)
        LOG_WARN_S << "The pointcloud filter has never been set!";
    if(!pc_set)
        LOG_WARN_S << "The pointcloud has never been set!";

    // Crop Box filter
    pcl::CropBox<pcl::PointXYZ> cb;
    cb.setInputCloud(cloud_input_p);
    cb.setMin(filter_box_min);
    cb.setMax(filter_box_max);
    cb.filter(*cloud_filtered_p);

    // Voxel grid filter
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud_filtered_p);
    sor.setLeafSize (leaf_size, leaf_size, leaf_size);
    sor.filter (*cloud_filtered_p);

    pc_filtered = 1;
}


int DEM::pointCloud2Mesh(bool use_filtered)
{
    if(!timestamp_set)
        LOG_WARN_S << "The timestamp has never been set!";

    if(!pc_set)
        LOG_WARN_S << "The pointcloud has never been set!";

    // Normal estimation*
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    if(pc_filtered && use_filtered)
    {
        if(cloud_filtered_p->size()<50)
        {
            LOG_DEBUG_S << "Not enough points to generate DEM";
            return -1;
        }
        tree->setInputCloud (cloud_filtered_p);
        n.setInputCloud (cloud_filtered_p);
    }
    else if(!use_filtered)
    {
        if(cloud_input_p->size()<50)
        {
            LOG_DEBUG_S << "Not enough points to generate DEM";
            return -1;
        }
        tree->setInputCloud (cloud_input_p);
        n.setInputCloud (cloud_input_p);
    }
    else
    {
        LOG_WARN_S << "You asked for a filtered PC to be put in a mesh but you forgot to call the filter!";
    }

    n.setSearchMethod (tree);
    n.setKSearch(k_points);
    n.compute (*normals);

    // normals should not contain the point normals + surface curvatures
    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);

    if(pc_filtered && use_filtered)
        pcl::concatenateFields (*cloud_filtered_p, *normals, *cloud_with_normals);
    else if(!use_filtered)
        pcl::concatenateFields (*cloud_input_p, *normals, *cloud_with_normals);
    else
        LOG_WARN_S << "You asked for a filtered PC to be put in a mesh but you forgot to call the filter!";

    // cloud_with_normals = cloud + normals
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

    // remove possible degenerate faces
    // ATTENTION assumes that faces are all triangles
    int count_degen = 0;
    for(int i=triangles.polygons.size()-1; i>= 0; i--)
    {
        // face is degenerate
        if(triangles.polygons[i].vertices[0] == triangles.polygons[i].vertices[1] ||
            triangles.polygons[i].vertices[0] == triangles.polygons[i].vertices[2] ||
            triangles.polygons[i].vertices[1] == triangles.polygons[i].vertices[2])
        {
                count_degen++;
                triangles.polygons.erase(triangles.polygons.begin()+i);
        }
    }
    LOG_DEBUG_S << "removed " << count_degen << " degenerated faces";


    // Additional vertex information
    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates();

    //Texture the mesh

    // test texture mapping
    pcl::TextureMapping<pcl::PointXYZ> tm;

    std::vector<std::string> color_frame_location_vect;
    color_frame_location_vect.push_back(color_frame_location_left);

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

    // Save mesh obj. Do we need 6 precision?
    mesh_location = constructProductPath("DEM",".obj");
    pcl::io::saveOBJFile (mesh_location, tex_mesh , 3);

    if(compression_enabled)
    {
        std::string command("gzip " + mesh_location);;
        system(command.c_str());
        mesh_location = mesh_location + ".gz";
    }

    return 0;
}


void DEM::mapTexture2MeshUVnew (pcl::TextureMesh &tex_mesh, pcl::TexMaterial &tex_material, std::vector<std::string> &tex_files)
{
    int nr_points = tex_mesh.cloud.width * tex_mesh.cloud.height;
    int point_size = static_cast<int> (tex_mesh.cloud.data.size ()) / nr_points;
    float x_, y_, z_;

    // texture coordinates for each mesh
    std::vector<std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > >texture_map;

    for (size_t m = 0; m < tex_mesh.tex_polygons.size (); ++m)
    {
        // texture coordinates for each mesh
        std::vector<Eigen::Vector2f> texture_map_tmp;

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
        LOG_WARN_S << "The timestamp has never been set!";

    cv::Mat tmp = cv::Mat(distance).reshape(0,height);
    tmp=tmp.mul(1000); //  go to mm distance
    tmp.convertTo(tmp,CV_16U); // save in uint16

    distance_frame_location = constructProductPath("DIST",".png");
    cv::imwrite(distance_frame_location, tmp);

    if(compression_enabled)
    {
        std::string command("gzip " + distance_frame_location);;
        system(command.c_str());
        distance_frame_location = distance_frame_location + ".gz";
    }
}

void DEM::savePointCloud(bool filtered)
{
    if(!timestamp_set)
        LOG_WARN_S << "The timestamp has never been set!";

    point_cloud_obj_location = constructProductPath("PC",".obj");

    pcl::PolygonMesh mesh;

    if(filtered)
        pcl::toPCLPointCloud2(*cloud_filtered_p, mesh.cloud);
    else
        pcl::toPCLPointCloud2(*cloud_input_p, mesh.cloud);

    pcl::io::saveOBJFile(point_cloud_obj_location, mesh);

    if(compression_enabled)
    {
        std::string command("gzip " + point_cloud_obj_location);;
        system(command.c_str());
        point_cloud_obj_location = point_cloud_obj_location + ".gz";
    }
}


std::string DEM::getMeshPath()
{
    return mesh_location;
}

std::string DEM::getPointCloudPath()
{
    return point_cloud_obj_location;
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
