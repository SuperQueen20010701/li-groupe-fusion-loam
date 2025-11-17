/// using the open3d icp to get an accurate initial pose guess
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

#include <open3d/Open3D.h>

using namespace open3d;
using namespace Eigen;
using namespace std;
using namespace tictoc;

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

struct KeyMeasurePose{
	Eigen::Vector3d translation_;
	Eigen::Quaterniond quaternion_;

    KeyMeasurePose():translation_(0.0, 0.0, 0.0),quaternion_(1.0, 0.0, 0.0, 0.0){};

    KeyMeasurePose(Eigen::Vector3d & trans_update , Eigen::Quaterniond  & quat_update) {
		translation_ = trans_update;
		quaternion_ = quat_update;
	}
};

struct IcpResult{
    Eigen::Matrix4d T_;
    double icp_rmse_;
    double fitness_;
    size_t inlier_p_cnt_;
    bool valid_;

};

struct EvaluateMetric{
    Eigen::Matrix4d T_final;
    double quality;
    bool accepted;
};

struct KeyframeMeasurement{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr edge_in;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr surf_in;
	KeyMeasurePose keyframe_pose;

	double time_keyframe;
	Measurement():edge_in(new pcl::PointCloud<pcl::PointXYZRGB>()),
	surf_in(new pcl::PointCloud<pcl::PointXYZRGB>()),
	odom_keyframe(Eigen::Isometry3d::Identity()),
	time_keyframe(0.0){};

}

struct Pose6D 
{
	double tx, ty, tz, rx, ry, rz;
	Pose6D():tx(0.0), ty(0.0), tz(0.0), rx(0.0), ry(0.0), rz(0.0){};
	Pose6D(double _tx, double _ty, double _tz, double _rx, double _ry, double _rz)
	:tx(_tx), ty(_ty), tz(_tz), rx(_rx), ry(_ry), rz(_rz){};
}

using PC2KD = PointCloudAdaptor<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>;
using kd_treee_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PC2KD>, PC2KD, 3>;

#endif

class OdomEstimationClass 
{

    public:
    	OdomEstimationClass();
    	
		void init(lidar::Lidar lidar_param, double map_resolution, double _k, double _theta, double _king, int icp_method_in, bool use_icp_in);	
		void initMapWithPoints(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_in);
		void updatePointsToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_in);
		void getMap(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& laserCloudMap);

		// base_link相对于map的位姿变换
		Eigen::Isometry3d odom;
		pcl::PointCloud<pcl::PointXYZI>::Ptr local_map;
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

		Eigen

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

#pragma region using the icp method as the initial pose guess
		bool use_icp_;
		int icp_method_;	
		Matrix4d init_trans_pose_ = Eigen::Matrix4d::Identity();
		IcpResult icp_result_surf_ , icp_result_corner_;
		std::vector<KeyframeMeasurement> keyframe_measurements_store;
		double keyframe_trans_accumulate = 1000000.0;
		double keyframe_trans_thres = 0.2;
        double keyframe_rot_thres = 0.1;

		EvaluateMetric evaluate_metric_icp = {.T_final = Eigen::Matrix4d::Identity(),
			.quality = 0.0, 
			.accepted = false};



		/// get local map based on init pose 
		void getLocalMap(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_local_map, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& corner_local_map,const Eigen::Matrix4d &initial_pose_matrix,
			float radius = 50.0);

		void Pcl2GeomtryPoint( const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_edge_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_surf_in,
								pcl::PointCloud<open3d::geometry::PointXYZ>::Ptr &pc_edge_out,pcl::PointCloud<open3d::geometry::PointXYZ>::Ptr &pc_surf_out);

		open3d::pipelines::registration::RegistrationResult ScanToLocalMapIcp(std::shared_ptr<open3d::geometry::PointCloud> &source,
									std::shared_ptr<open3d::geometry::PointCloud> &target,
									const Eigen::Matrix4d &init_pose,
									const open3d::pipelines::registration::ICPConvergenceCriteria &criteria,
									const std::string & regis_type);

		size_t getInlinerIcpPointCount(const open3d::geometry::PointCloud & source ,
			const open3d::geometry::PointCloud & target ,
			 const Eigen::Matrix4d & trans_mat , double max_corr_dist);

		bool checkICPResult(const pipelines::registration::RegistrationResult & icp_result_corner ,
				const pipelines::registration::RegistrationResult & icp_result_surf);

		void calculateIcpQuality(IcpResult & icp_result_in);

		bool SaveKeyframeRadius(Eigen::Isometry3d & curr_odom ,Eigen::Isometry3d & last_odom);

		Pose6D getTransformation(Eigen::Isometry3d & curr_odom ,Eigen::Isometry3d & last_odom);
#pragma endregion

		//function
		void addEdgeCostFactor(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& map_in, ceres::Problem& problem, ceres::LossFunction *loss_function);
		void addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& map_in, ceres::Problem& problem, ceres::LossFunction *loss_function);
		void addPointsToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampledEdgeCloud, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampledSurfCloud);
		void pointAssociateToMap(pcl::PointXYZRGB const *const pi, pcl::PointXYZRGB *const po);
		void downSamplingToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_pc_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_pc_out, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_pc_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_pc_out);
		void updatePose();
};

#endif // _ODOM_ESTIMATION_CLASS_H_

