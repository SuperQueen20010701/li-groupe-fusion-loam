// Author of FLOAM: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
#ifndef _ODOM_ESTIMATION_CLASS_H_
#define _ODOM_ESTIMATION_CLASS_H_

//std lib
#include <string>
#include <math.h>
#include <vector>

//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>

//ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

//eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

//LOCAL LIB
#include "lidar.h"
#include "lidarOptimization.h"
#include <ros/ros.h>

#include <algorithm>
#include <numeric>
#include "tic_toc.h"
#ifdef NANOFLANN
#include <scancontext/nanoflann.hpp>


template <typename Derived>
struct PointCloudAdaptor
{

	using PC2KD = PointCloudAdaptor<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>;
	using kd_treee_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PC2KD>, PC2KD, 3>;

    const Derived& obj;  //!< A const ref to the data set origin

    /// The constructor that sets the data set source
    // PointCloudAdaptor(const Derived& obj_) : obj(obj_) {}

	kd_treee_t* index; //! The kd-tree index for the user to call its methods as usual with any other FLANN index.

	/// Constructor: takes a const ref to the vector of vectors object with the data points
	PointCloudAdaptor(const Derived &obj_) : obj(obj_)
	{
		index = new kd_treee_t( 3, *this /* adaptor */, {40} );
		index->buildIndex();
	}

	~PointCloudAdaptor() {
		delete index;
	}

	// inline void setInputCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloudPoints){
	// 	delete index;
	// 	index = new kd_treee_t( 3, *this /* adaptor */, {30} );
	// 	obj = cloudPoints;
	// 	index->buildIndex();
	// }

	inline void nearestKSearch(pcl::PointXYZRGB &point, int searchNum, std::vector<size_t> &pointSearchInd, std::vector<float> &pointSearchSqDis){
		pointSearchInd.resize(searchNum);
		pointSearchSqDis.resize(searchNum);
        const float query_pt[3] = {point.x, point.y, point.z};
		index->knnSearch(query_pt, searchNum, &pointSearchInd[0], &pointSearchSqDis[0]);
	}

    /// CRTP helper method
    inline const Derived& derived() const { return obj; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const
    {
        return derived()->points.size();
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return derived()->points[idx].x;
        else if (dim == 1)
            return derived()->points[idx].y;
        else
            return derived()->points[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const
    {
        return false;
    }

};  // end of PointCloudAdaptor


using PC2KD = PointCloudAdaptor<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>;
using kd_treee_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PC2KD>, PC2KD, 3>;

#endif

struct Corner_Corr{
	Eigen::Vector3d pc_;
	Eigen::Vector3d pa_;
	Eigen::Vector3d pb_;

	double wei_c;
};

struct Surf_Corr{
	Eigen::Vector3d pc_;
	Eigen::Vector3d norm ;
	double neg ;
	int rgb_;
	double wei_s ;
};

class OdomEstimationClass 
{

    public:
    	OdomEstimationClass();
    	
		void init(lidar::Lidar lidar_param, double map_resolution, double _k, double _theta, double _king);	
		void initMapWithPoints(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_in);
		void updatePointsToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_in);
		void getMap(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& laserCloudMap);

		// base_link相对于map的位姿变换
		Eigen::Isometry3d odom;
		// 线特征点map
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudCornerMap;
		// 面特征点map
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr laserCloudSurfMap;
	private:
		// rx ry rz x y z
		double paramEuler[6] = {0, 0, 0, 0, 0, 0};
		double k_;
		double theta;
		double king;
		Eigen::Quaterniond q_w_curr;
		Eigen::Vector3d t_w_curr;

		Eigen::Isometry3d last_odom;

		//kd-tree
		#ifdef NANOFLANN
		// 线特征地图
		std::unique_ptr<PC2KD> kdtreeEdgeMap;
		// 面特征地图
		std::unique_ptr<PC2KD> kdtreeSurfMap;
		#else
		// 线特征地图
		pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr kdtreeEdgeMap;
		// 面特征地图
		pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr kdtreeSurfMap;
		#endif

		//points downsampling before add to map
		pcl::VoxelGrid<pcl::PointXYZRGB> downSizeFilterEdge;
		pcl::VoxelGrid<pcl::PointXYZRGB> downSizeFilterSurf;

		//local map
		pcl::CropBox<pcl::PointXYZRGB> cropBoxFilter;

		//optimization count，限制优化的次数
		int optimization_count;

		//function
		void addEdgeCostFactor(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& map_in);
		void addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& map_in);
		void addPointsToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampledEdgeCloud, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampledSurfCloud);
		void pointAssociateToMap(pcl::PointXYZRGB const *const pi, pcl::PointXYZRGB *const po);
		void downSamplingToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_pc_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_pc_out, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_pc_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_pc_out);
		void updatePose();

		Eigen::Matrix3d R_init;
		Eigen::Vector3d t_init = Eigen::Vector3d::Identity();

		Eigen::Isometry3d T_init = Eigen::Isometry3d::Identity();

		std::vector<Corner_Corr> cor_corrs_vec ;

		std::vector<Surf_Corr> surf_corrs_vec ;

		std::vector<size_t> surf_sel ,corner_sel;
		const size_t max_select = 2000;

		double RobustMADEstimation(const std::vector<double> res_vec_in);

		std::vector<size_t> selectTopNCorner(size_t N, const std::vector<Corner_Corr> & res_cor);

		std::vector<size_t> selectTopNSurf(size_t N, const std::vector<Surf_Corr> & res_surf);

		Eigen::Isometry3d ParamToIso(const double p[6]);

		double median(std::vector<double>& v);

};

#endif // _ODOM_ESTIMATION_CLASS_H_