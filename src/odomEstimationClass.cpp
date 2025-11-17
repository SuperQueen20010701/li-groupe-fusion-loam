// Author of FLOAM: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#include "odomEstimationClass.h"

struct cloud_point_index_idx 
{
    unsigned int idx;
    unsigned int cloud_point_index;

    cloud_point_index_idx (unsigned int idx_, unsigned int cloud_point_index_) : idx (idx_), cloud_point_index (cloud_point_index_) {}
    bool operator < (const cloud_point_index_idx &p) const { return (idx < p.idx); }
};

void extractstablepoint(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, int k, float theta, int king){
    std::vector<int> index;
    for(int i = 0; i < input->points.size(); i++){
        if(input->points[i].g < input->points[i].r * theta && input->points[i].r > k && input->points[i].g <king + 1)
            continue; 
        index.push_back(i);
    }
    boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(index);
    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    // Extract the inliers
    extract.setInputCloud (input);
    extract.setIndices (index_ptr);
    extract.setNegative (false);//如果设为true,可以提取指定index之外的点云
    extract.filter (*input);
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbds (pcl::PointCloud<pcl::PointXYZRGB>::Ptr input, float dsleaf, int min_points_per_voxel_=0){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);
    output->height = 1;        
	output->is_dense = true; 

	Eigen::Vector4f min_p, max_p;
    Eigen::Vector4i min_b_, max_b_, div_b_, divb_mul_;
    pcl::getMinMax3D<pcl::PointXYZRGB> (*input, min_p, max_p); ///计算出最小点和最大点

    min_b_[0] = static_cast<int> (floor (min_p[0] / dsleaf));// 按照体素大小划分为索引
	max_b_[0] = static_cast<int> (floor (max_p[0] / dsleaf));
	min_b_[1] = static_cast<int> (floor (min_p[1] / dsleaf));
	max_b_[1] = static_cast<int> (floor (max_p[1] / dsleaf));
	min_b_[2] = static_cast<int> (floor (min_p[2] / dsleaf));
	max_b_[2] = static_cast<int> (floor (max_p[2] / dsleaf));

    div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones ();
	div_b_[3] = 0;
	divb_mul_ = Eigen::Vector4i (1, div_b_[0], div_b_[0] * div_b_[1], 0);  ///用作一维索引展开的step 系数

    std::vector<cloud_point_index_idx> index_vector;
	index_vector.reserve (input->points.size ());

    for (int i = 0; i < input->points.size(); i++){
        int ijk0 = static_cast<int> (floor (input->points[i].x / dsleaf) - static_cast<float> (min_b_[0]));
        int ijk1 = static_cast<int> (floor (input->points[i].y / dsleaf) - static_cast<float> (min_b_[1]));
        int ijk2 = static_cast<int> (floor (input->points[i].z / dsleaf) - static_cast<float> (min_b_[2]));

        // Compute the centroid leaf index
        int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];
        index_vector.push_back (cloud_point_index_idx (static_cast<unsigned int> (idx), i));
    }

	// Second pass: sort the index_vector vector using value representing target cell as index
	// in effect all points belonging to the same output cell will be next to each other
	std::sort (index_vector.begin (), index_vector.end (), std::less<cloud_point_index_idx> ());

	// Third pass: count output cells
	// we need to skip all the same, adjacenent idx values
	unsigned int total = 0;
	unsigned int index = 0;
	// first_and_last_indices_vector[i] represents the index in index_vector of the first point in
	// index_vector belonging to the voxel which corresponds to the i-th output point,
	// and of the first point not belonging to.
	std::vector<std::pair<unsigned int, unsigned int> > first_and_last_indices_vector;
	// Worst case size
    /// 构建体素地图
	first_and_last_indices_vector.reserve (index_vector.size ());
	while (index < index_vector.size ()) 
	{
		unsigned int i = index + 1;
		while (i < index_vector.size () && index_vector[i].idx == index_vector[index].idx) 
		    ++i;
		if (i - index >= min_points_per_voxel_){
    		++total;
	     	first_and_last_indices_vector.push_back (std::pair<unsigned int, unsigned int> (index, i));
		}
		index = i;
	}

	// Fourth pass: compute centroids, insert them into their final position
	output->points.resize (total);
	index = 0;
	for (unsigned int cp = 0; cp < first_and_last_indices_vector.size (); ++cp){
		// calculate centroid - sum values from all input points, that have the same idx value in index_vector array
		unsigned int first_index = first_and_last_indices_vector[cp].first;
		unsigned int last_index = first_and_last_indices_vector[cp].second;

		
        Eigen::Vector4f centroid (Eigen::Vector4f::Zero ());

        int r_max = -1;
        float g_max = -1; 
        for (unsigned int li = first_index; li < last_index; ++li){
            centroid += input->points[index_vector[li].cloud_point_index].getVector4fMap ();
            if(input->points[index_vector[li].cloud_point_index].r > r_max){
                r_max = input->points[index_vector[li].cloud_point_index].r;
                // centroid = input->points[index_vector[li].cloud_point_index].getVector4fMap ();
            }
            if(input->points[index_vector[li].cloud_point_index].g > g_max){
                g_max = input->points[index_vector[li].cloud_point_index].g;
            }
        }
        centroid /= static_cast<float> (last_index - first_index);
        output->points[index].getVector4fMap () = centroid;
        output->points[index].r = r_max;
        output->points[index].g = g_max;

		++index;
	}
	output->width = static_cast<uint32_t> (output->points.size ());
	return output;
}

void OdomEstimationClass::init(lidar::Lidar lidar_param, double map_resolution, double _k, double _theta, double _king, int icp_method_in, bool use_icp_in){

    k_ = _k;
    theta = _theta;
    king = _king;

    ROS_WARN("%f %f %f\n\n", k_, theta, king);

    //init local map
    laserCloudCornerMap = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    laserCloudSurfMap = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());

    //downsampling size
    downSizeFilterEdge.setLeafSize(map_resolution, map_resolution, map_resolution);
    downSizeFilterSurf.setLeafSize(map_resolution * 2, map_resolution * 2, map_resolution * 2);

    //kd-tree
    #ifdef NANOFLANN
    #else
    kdtreeEdgeMap = pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZRGB>());
    kdtreeSurfMap = pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZRGB>());
    #endif

    odom = Eigen::Isometry3d::Identity();
    last_odom = Eigen::Isometry3d::Identity();
    optimization_count=2;

    icp_method_ = icp_method_in;
    use_icp_ = use_icp_in;

    icp_result_surf_ = {.T_ = Eigen::Matrix4d::Identity(),
        .icp_rmse_ = 0.0,
        .fitness_ = 0.0,
        .inlier_p_cnt_ = 0,
        .valid_ = true};

    icp_result_corner_ = {.T_ = Eigen::Matrix4d::Identity(),
        .icp_rmse_ = 0.0,
        .fitness_ = 0.0,
        .inlier_p_cnt_ = 0,
        .valid_ = true};

}

void OdomEstimationClass::initMapWithPoints(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_in){
    // 添加特征点到Map
    *laserCloudCornerMap += *edge_in;
    *laserCloudSurfMap += *surf_in;
    optimization_count=12;
}
/***
add the icp matching method 
get the initial pose guess by icp method
ipdate the local map 
publish local map 
 ***/

void OdomEstimationClass::updatePointsToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_in){

    // 为什么需要把优化次数慢慢降下来？？
    if(optimization_count>2)
        optimization_count--;

    // 利用前一次位姿变换来预测当前位姿变换 odom * last_odom.inverse 相当于odom相对于last_odom的增量
    Eigen::Isometry3d odom_prediction = odom * (last_odom.inverse() * odom);
    last_odom = odom;
    odom = odom_prediction;

    bool keyframe_satisfy = false;
    // 储存当前点云的位姿
    q_w_curr = Eigen::Quaterniond(odom.rotation());
    t_w_curr = odom.translation();

    ///whether to save the keyframe
    if(SaveKeyframeRadius(odom, last_odom))
    {
        keyframe_satisfy = true;
    }

#pragma region using the icp method as the initial pose guess
    if(use_icp_){
        init_trans_pose_.topLeftCorner<3,3>() = q_w_curr.toRotationMatrix();
        init_trans_pose_.topRightCorner<3,1>() = t_w_curr;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr surf_local_map(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr corner_local_map(new pcl::PointCloud<pcl::PointXYZRGB>);

        getLocalMap(surf_local_map, corner_local_map, init_trans_pose_, 80.0);

        if(!surf_local_map.size() >100 && !corner_local_map.size() >100){
            
            ///get the local map 
            std::shared_ptr<open3d::geometry::PointCloud> pc_edge_map_out(new open3d::geometry::PointCloud);
            std::shared_ptr<open3d::geometry::PointCloud> pc_surf_map_out(new open3d::geometry::PointCloud);
            Pcl2GeomtryPoint(surf_local_map, corner_local_map, pc_edge_map_out, pc_surf_map_out);
            /// get curr geometry point cloud
            std::shared_ptr<open3d::geometry::PointCloud> pc_edge_scan_out(new open3d::geometry::PointCloud);
            std::shared_ptr<open3d::geometry::PointCloud> pc_surf_scan_out(new open3d::geometry::PointCloud);
            Pcl2GeomtryPoint(edge_in, surf_in, pc_edge_scan_out, pc_surf_scan_out);
            /// 利用open3d icp进行匹配
            pipelines::registration::RegistrationResult icp_result_edge ;
            pipelines::registration::RegistrationResult icp_result_surf;

            auto criteria = pipelines::registration::ICPConvergenceCriteria(10);
            
            icp_result_edge = ScanToLocalMapIcp(pc_edge_scan_out, pc_edge_map_out, init_trans_pose_, criteria, "corner_icp");
            icp_result_surf = ScanToLocalMapIcp(pc_surf_scan_out, pc_surf_map_out, init_trans_pose_, criteria, "surf_icp");

            /// 检查icp的结果
            if(checkICPResult(icp_result_edge, icp_result_surf))
            {
                if(evaluate_metric_icp.T_final != Eigen::Matrix4d::Identity() &&
                evaluate_metric_icp.accepted == true)
                {
                    q_w_curr =Eigen::Quaterniond(evaluate_metric_icp.T_final.block<3,3>(0,0)).normalized(); // using the icp refined pose
                    t_w_curr = evaluate_metric_icp.T_final.block<3,1>(0,3);
                    init_trans_pose_ = evaluate_metric_icp.T_final;
                    ROS_INFO("update the initial guess trans pose using icp");
                }
            }else{
                ROS_WARN("icp failed, not meet the matching requirements");
                evaluate_metric_icp.accepted = false;
                evaluate_metric_icp.quality = 0.0;
                evaluate_metric_icp.T_final = init_trans_pose_;
            }
        }else{
            ROS_WARN("WARN !! not enough points in  map to associate");
        }
    }
#pragma endregion

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampledEdgeCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampledSurfCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    // 下采样两个特征点云
    downSamplingToMap(edge_in,downsampledEdgeCloud,surf_in,downsampledSurfCloud);

    if(laserCloudCornerMap->points.size()>10 && laserCloudSurfMap->points.size()>50){
        #ifdef NANOFLANN
        kdtreeEdgeMap.reset();
        kdtreeEdgeMap = std::make_unique<PC2KD>(laserCloudCornerMap);
        kdtreeSurfMap.reset();
        kdtreeSurfMap = std::make_unique<PC2KD>(laserCloudSurfMap);
        #else
        kdtreeEdgeMap->setInputCloud(laserCloudCornerMap);
        kdtreeSurfMap->setInputCloud(laserCloudSurfMap);
        #endif

        // 优化optimization_count轮（每次会对parameter在前一次基础上进行优化）
        for (int iterCount = 0; iterCount < optimization_count; iterCount++){
            {
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::Problem::Options problem_options;
                // 优化问题
                ceres::Problem problem(problem_options);

                // 设置需要优化的参数
                problem.AddParameterBlock(paramEuler, 6);
                
                // 添加面特征项
                addSurfCostFactor(downsampledSurfCloud,laserCloudSurfMap,problem,loss_function);

                // 优化参数设置
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 4;
                options.minimizer_progress_to_stdout = false;
                options.check_gradients = false;
                options.gradient_check_relative_precision = 1e-4;
                ceres::Solver::Summary summary;

                // solve 会修改paramEuler
                ceres::Solve(options, &problem, &summary);
                updatePose();
            }
        // }
        // for (int iterCount = 0; iterCount < optimization_count; iterCount++){
            {
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::Problem::Options problem_options;
                // 优化问题
                ceres::Problem problem(problem_options);

                // 设置需要优化的参数
                problem.AddParameterBlock(paramEuler, 6);
                
                // 添加线特征项
                addEdgeCostFactor(downsampledEdgeCloud,laserCloudCornerMap,problem,loss_function);

                // 优化参数设置
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 4;
                options.minimizer_progress_to_stdout = false;
                options.check_gradients = false;
                options.gradient_check_relative_precision = 1e-4;
                ceres::Solver::Summary summary;

                // solve 会修改paramEuler
                ceres::Solve(options, &problem, &summary);
                updatePose();
            }

        }
    }else{
        printf("not enough points in map to associate, map error");
    }
    // 更新odom坐标，为下一次prediction作准备
    odom = Eigen::Isometry3d::Identity();
    odom.linear() = q_w_curr.toRotationMatrix();
    odom.translation() = t_w_curr;
    // 添加特征点到map中

    if(keyframe_satisfy)/// 存放关键帧
    {
        addPointsToMap(downsampledEdgeCloud,downsampledSurfCloud);
        ROS_INFO("更新局部地图");
    }
   
}

// 根据当前欧拉角的参数更新旋转四元数和平移矩阵
void OdomEstimationClass::updatePose(){
    // 围绕x y z轴旋转的角度，对应pitch yaw roll
    double rx = paramEuler[0], ry = paramEuler[1], rz = paramEuler[2];
    // x y z方向平移的距离
    double tx = paramEuler[3], ty = paramEuler[4], tz = paramEuler[5];
    // Z * Y * X
    Eigen::Matrix3d R = (
        Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()) * 
        Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) * 
        Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX())
        ).toRotationMatrix();
    Eigen::Vector3d T;
    T <<    tx, ty, tz;

    q_w_curr = Eigen::Quaterniond(R);
    t_w_curr = T;
}

// 通过当前变换矩阵，将pi变换到po
void OdomEstimationClass::pointAssociateToMap(pcl::PointXYZRGB const *const pi, pcl::PointXYZRGB *const po)
{
    Eigen::Isometry3d isoMat;
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->b = pi->b;
    //po->b = 1.0;
}

void OdomEstimationClass::downSamplingToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_pc_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_pc_out, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_pc_in, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_pc_out){
    downSizeFilterEdge.setInputCloud(edge_pc_in);
    downSizeFilterEdge.filter(*edge_pc_out);
    downSizeFilterSurf.setInputCloud(surf_pc_in);
    downSizeFilterSurf.filter(*surf_pc_out);    
}

// 输入：特征点点云，特征点点云地图，优化问题，loss函数
void OdomEstimationClass::addEdgeCostFactor(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& map_in, ceres::Problem& problem, ceres::LossFunction *loss_function){
    int corner_num=0;
    // 遍历输入的特征点
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    { 
        // 经过当前待优化的变换矩阵变换得到的点坐标
        pcl::PointXYZRGB point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);

        // k临近查找到的点的index
        #ifdef NANOFLANN
        std::vector<size_t> pointSearchInd;
        #else
        std::vector<int> pointSearchInd;
        #endif
        // k临近查找到的点的距离
        std::vector<float> pointSearchSqDis;
        // 搜索point_temp周围的5个点
        kdtreeEdgeMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis); 
        // 如果最远点距离也比较近
        if (pointSearchSqDis[4] < 1.0)
        {
            // 依次储存5个最近点
            std::vector<Eigen::Vector3d> nearCorners;
            // 储存5个点的重心坐标
            Eigen::Vector3d center(0, 0, 0);
            // 取得5个点的重心
            for (int j = 0; j < 5; j++)
            {
                Eigen::Vector3d tmp(map_in->points[pointSearchInd[j]].x,
                                    map_in->points[pointSearchInd[j]].y,
                                    map_in->points[pointSearchInd[j]].z);
                center = center + tmp;
                nearCorners.push_back(tmp);
            }
            center = center / 5.0;

            // 5个点协方差矩阵
            // https://njuferret.github.io/2019/07/28/2019-07-28_geometric-interpretation-covariance-matrix/
            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
            for (int j = 0; j < 5; j++)
            {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

            // 5个点构成的线的方向
            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
            // 未经变换的当前点，和point_temp不同
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
            // 如果这5个点确实近似构成一条直线
            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
            { 

                float observe = (map_in->points[pointSearchInd[0]].g + 
                                map_in->points[pointSearchInd[1]].g +
                                map_in->points[pointSearchInd[2]].g +
                                map_in->points[pointSearchInd[3]].g +
                                map_in->points[pointSearchInd[4]].g ) / 5.0 + 1;
                float round = (map_in->points[pointSearchInd[0]].r + 
                                map_in->points[pointSearchInd[1]].r +
                                map_in->points[pointSearchInd[2]].r +
                                map_in->points[pointSearchInd[3]].r +
                                map_in->points[pointSearchInd[4]].r ) / 5.0;
                for (int j = 0; j < 5; j++){   
                    map_in->points[pointSearchInd[j]].g = std::min(255, map_in->points[pointSearchInd[j]].g + 1); /// 观测次数
                }
                
                if(observe < round * theta && round > k_ && observe < king){
                    assert(0);
                    continue;
                }

                // 5个点的中心作为在直线上的点
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b;
                // 在直线上找2个点
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;

                // 添加一个边的误差项
                ceres::CostFunction *cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b);  
                problem.AddResidualBlock(cost_function, loss_function, paramEuler);
                corner_num++;   
            }                           
        }
    }
    if(corner_num<20){
        printf("not enough correct points");
    }

}

void OdomEstimationClass::addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& map_in, ceres::Problem& problem, ceres::LossFunction *loss_function){
    int surf_num=0;
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {
        pcl::PointXYZRGB point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);
        #ifdef NANOFLANN
        std::vector<size_t> pointSearchInd;
        #else
        std::vector<int> pointSearchInd;
        #endif
        std::vector<float> pointSearchSqDis;
        // 在经过变换之后的面特征地图中查找最近的5个点
        kdtreeSurfMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);

        // 每一行储存一个点的坐标
        Eigen::Matrix<double, 5, 3> matA0;
        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
        if (pointSearchSqDis[4] < 1.0)
        {
            // 向matA0中添加点
            for (int j = 0; j < 5; j++)
            {
                matA0(j, 0) = map_in->points[pointSearchInd[j]].x;
                matA0(j, 1) = map_in->points[pointSearchInd[j]].y;
                matA0(j, 2) = map_in->points[pointSearchInd[j]].z;
            }
            // find the norm of plane，解方程matA0 . norm = matB0， 此时norm和5个点任意一个点的内积接近-1，norm是平面法向量
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
            // norm的长度倒数
            double negative_OA_dot_norm = 1 / norm.norm();
            // 单位化
            norm.normalize();

            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                // if OX * n > 0.2, then plane is not fit well，平面法向量和平面点的内积，如果严格垂直的话内积为0
                if (fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
                         norm(1) * map_in->points[pointSearchInd[j]].y +
                         norm(2) * map_in->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                {
                    planeValid = false;
                    break;
                }
            }
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
            // 如果平面存在
            if (planeValid)
            {
                float observe = (map_in->points[pointSearchInd[0]].g + 
                                map_in->points[pointSearchInd[1]].g +
                                map_in->points[pointSearchInd[2]].g +
                                map_in->points[pointSearchInd[3]].g +
                                map_in->points[pointSearchInd[4]].g ) / 5.0 + 1;
                float round = (map_in->points[pointSearchInd[0]].r + 
                                map_in->points[pointSearchInd[1]].r +
                                map_in->points[pointSearchInd[2]].r +
                                map_in->points[pointSearchInd[3]].r +
                                map_in->points[pointSearchInd[4]].r ) / 5.0;
                for (int j = 0; j < 5; j++){   
                    map_in->points[pointSearchInd[j]].g = std::min(255, map_in->points[pointSearchInd[j]].g + 1);
                }
                if(observe < round * theta && round > k_ && observe < king){
                    assert(0);
                    continue;
                }
                // 添加误差项
                ceres::CostFunction *cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, pc_in->points[i].b);    
                problem.AddResidualBlock(cost_function, loss_function, paramEuler);

                surf_num++;
            }
        }

    }
    if(surf_num<20){
        printf("not enough correct points");
    }

}

// 添加特征点到edge和surf地图中
void OdomEstimationClass::addPointsToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampledEdgeCloud, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampledSurfCloud){

    // 将cloud中经过优化完成的变换矩阵的变换，然后添加到对应的地图中
    if(!icp_result_corner_.valid_)
    {
        for (int i = 0; i < (int)downsampledEdgeCloud->points.size(); i++)
        {
            pcl::PointXYZRGB point_temp;
            pointAssociateToMap(&downsampledEdgeCloud->points[i], &point_temp);
            laserCloudCornerMap->push_back(point_temp); 
        }
    }


    if(!icp_result_surf_.valid_)
    {
        for (int i = 0; i < (int)downsampledSurfCloud->points.size(); i++)
        {
            pcl::PointXYZRGB point_temp;
            pointAssociateToMap(&downsampledSurfCloud->points[i], &point_temp);
            laserCloudSurfMap->push_back(point_temp);
        }
    }

    // 更新local map大小
    double x_min = +odom.translation().x()-100;
    double y_min = +odom.translation().y()-100;
    double z_min = +odom.translation().z()-100;
    double x_max = +odom.translation().x()+100;
    double y_max = +odom.translation().y()+100;
    double z_max = +odom.translation().z()+100;
    
    //ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max, z_max);
    cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
    cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
    cropBoxFilter.setNegative(false);    

    // 重新选取特征点云地图，以防特征点云地图越来越大
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpCorner(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpSurf(new pcl::PointCloud<pcl::PointXYZRGB>());
    cropBoxFilter.setInputCloud(laserCloudSurfMap);
    cropBoxFilter.filter(*tmpSurf);
    cropBoxFilter.setInputCloud(laserCloudCornerMap);
    cropBoxFilter.filter(*tmpCorner);

    laserCloudSurfMap = rgbds(tmpSurf, 0.4*2 );
    laserCloudCornerMap = rgbds(tmpCorner, 0.4);
    extractstablepoint(laserCloudSurfMap, k_, theta, king);
    extractstablepoint(laserCloudCornerMap, k_, theta, king);

    for(int i = 0;i < laserCloudSurfMap->points.size(); i++){
        if(laserCloudSurfMap->points[i].r > 250)
            laserCloudSurfMap->points[i].r = 255;
        else
            laserCloudSurfMap->points[i].r +=2;
    }
        
    for(int i = 0;i < laserCloudCornerMap->points.size(); i++)
        if(laserCloudCornerMap->points[i].r > 250)
            laserCloudCornerMap->points[i].r = 255;
        else
            laserCloudCornerMap->points[i].r +=2;
    
    local_map = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
    if(laserCloudCornerMap->size() > 0 && laserCloudSurfMap->size() > 0)
    {
        for(int i = 0;i < laserCloudCornerMap->points.size(); i++)
        {
            pcl::PointXYZI p_t ;
            p_t.x = laserCloudCornerMap->points[i].x;
            p_t.y = laserCloudCornerMap->points[i].y;
            p_t.z = laserCloudCornerMap->points[i].z;
            p_t.intensity = laserCloudCornerMap->points[i].r;
            local_map->push_back(p_t);
        }
        for(int i = 0;i < laserCloudSurfMap->points.size(); i++)
        {
            pcl::PointXYZI p_t ;
            p_t.x = laserCloudSurfMap->points[i].x;
            p_t.y = laserCloudSurfMap->points[i].y;
            p_t.z = laserCloudSurfMap->points[i].z;
            p_t.intensity = laserCloudSurfMap->points[i].r; // 反应过期的程度
            local_map->push_back(p_t);
        }
    }

}

void OdomEstimationClass::getMap(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& laserCloudMap){
    
    *laserCloudMap += *laserCloudSurfMap;
    *laserCloudMap += *laserCloudCornerMap;
}

OdomEstimationClass::OdomEstimationClass(){

}

void OdomEstimationClass::getLocalMap(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_local_map, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& corner_local_map,
    const Eigen::Matrix4d &initial_pose_matrix,float radius)
{
    surf_local_map->clear();
    corner_local_map->clear();

    Eigen::Matrix3f curr_rot = initial_pose_matrix.block<3, 3>(0, 0).cast<float>();
    Eigen::Vector3f curr_trans = initial_pose_matrix.block<3, 1>(0, 3).cast<float>();

    Eigen::Affine3f curr_pos;
    curr_pos.linear() = rotation;
    curr_pos.translation() = translation;

    Eigen::Vector4f min_pt(-radius, -radius, -std::numeric_limits<float>::infinity(), 1.0);
    Eigen::Vector4f max_pt(radius, radius, std::numeric_limits<float>::infinity(), 1.0);

    /// 利用distance 对global map进行切割 得到local map
    pcl::CropBox<pcl::PointXYZRGB> box_filter;
    box_filter.setInputCloud(laserCloudCornerMap);
    box_filter.setMin(min_pt);
    box_filter.setMax(max_pt);
    box_filter.setTranslation(pose_trans.translation());
    box_filter.setRotation(pose_trans.rotation().eulerAngles(0, 1, 2));
    box_filter.filter(*corner_local_map);

    /// for the local map of the local surf map ;
    box_filter.setInputCloud(laserCloudSurfMap);
    box_filter.setMin(min_pt);
    box_filter.setMax(max_pt);
    box_filter.setTranslation(pose_trans.translation());
    box_filter.setRotation(pose_trans.rotation().eulerAngles(0, 1, 2));
    box_filter.filter(*surf_local_map);
}

void OdomEstimationClass::Pcl2GeomtryPoint( const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_edge_in, 
                                            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pc_surf_in,
                                            std::shared_ptr<open3d::geometry::PointCloud> &pc_edge_out,
                                            std::shared_ptr<open3d::geometry::PointCloud> &pc_surf_out)
{

    pc_edge_out->points_.resize(pc_in_edge->size());
    pc_surf_out->points_.resize(pc_in_surf->size());
    /// for edge
    for(int i = 0 ; i < pc_in_edge->size() ; ++ i)
    {
        pc_edge_out->points_[i][0] = pc_in_edge->points.at(i).x;
        pc_edge_out->points_[i][1] = pc_in_edge->points.at(i).y;
        pc_edge_out->points_[i][2] = pc_in_edge->points.at(i).z;
    }
    /// for surf
    for(int i = 0 ; i < pc_in_surf->size() ; ++ i)
    {
        pc_surf_out->points_[i][0] = pc_in_surf->points.at(i).x;
        pc_surf_out->points_[i][1] = pc_in_surf->points.at(i).y;
        pc_surf_out->points_[i][2] = pc_in_surf->points.at(i).z;
    }

}

open3d::pipelines::registration::RegistrationResult OdomEstimationClass::ScanToLocalMapIcp(std::shared_ptr<open3d::geometry::PointCloud> &source,
    std::shared_ptr<open3d::geometry::PointCloud> &target,
    const Eigen::Matrix4d &init_pose,
    const open3d::pipelines::registration::ICPConvergenceCriteria &criteria,
    const std::string & regis_type)
{
    using namespace open3d;

    pipelines::registration::RegistrationResult icp_result;

    double max_correspondence_distance = 2.0;

    switch(icp_method_){
        case 1:
        {
            /// Generalized ICP
            std::cout << "[INFO] Running generalized ICP..." << std::endl;
            target->EstimateNormals(geometry::KDTreeSearchParamHybrid(1.0, 10));
            icp_result = pipelines::registration::RegistrationGeneralizedICP(
                    *source, *target, max_correspondence_distance, init_pose.cast<double>(),
                    pipelines::registration::TransformationEstimationForGeneralizedICP(), criteria);

            if(regis_type == "corner_icp")
            {
                icp_result_corner_.inlier_p_cnt_ = getInlinerIcpPointCount(*source, *target, icp_result.transformation_, 1.0);
                ROS_INFO("%s:matched point size is: %d \n","icp_corner",icp_result_corner_.inlier_p_cnt_);
            }else if(regis_type == "surf_icp")
            {
                icp_result_surf_.inlier_p_cnt_ = getInlinerIcpPointCount(*source, *target, icp_result.transformation_, 1.0);
                ROS_INFO("%s:matched point size is: %d \n","icp_surf",icp_result_surf_.inlier_p_cnt_);
            }
        }
        case 2:
        {
            ///  Multi-Scale ICP
            std::cout << "[INFO] Using multi-scale ICP!" << std::endl;
            target->EstimateNormals(geometry::KDTreeSearchParamHybrid(1.0, 10));

            /// tensor
            auto source_tensor = open3d::t::geometry::PointCloud::FromLegacy(*source).VoxelDownSample(0.5);
            auto target_tensor = open3d::t::geometry::PointCloud::FromLegacy(*target).VoxelDownSample(0.5);

            open3d::core::Tensor init_trans_matrix = open3d::core::eigen_converter::EigenMatrixToTensor(init_pose.cast<double>());
            std::vector<double> voxel_sizes = {0.5, 0.3, 0.2};
            std::vector<double> max_correspondence_distances = {5.0, 1.0, 0.5};

            ///多层级收敛原则
            std::vector<t::pipelines::registration::ICPConvergenceCriteria> multi_criteria_coarse_to_refine = {
                    t::pipelines::registration::ICPConvergenceCriteria(0.01, 0.01, 15),
                    t::pipelines::registration::ICPConvergenceCriteria(0.001, 0.001, 10),
                    t::pipelines::registration::ICPConvergenceCriteria(0.00001, 0.00001, 5)
            };

            auto multi_icp_result = t::pipelines::registration::MultiScaleICP(
                    source_tensor,target_tensor,voxel_sizes,multi_criteria_coarse_to_refine,
                    max_correspondence_distances,init_trans_matrix,
                    t::pipelines::registration::TransformationEstimationPointToPlane());
            icp_result.transformation_ = open3d::core::eigen_converter::TensorToEigenMatrixXd(multi_icp_result.transformation_);
            icp_result.inlier_rmse_ = multi_icp_result.inlier_rmse_;
            icp_result.fitness_ = multi_icp_result.fitness_;
            ///计算icp内点
            if(regis_type == "corner_icp")
            {
                icp_corner.inlier_p_cnt_ = getInlinerIcpPointCount(*source,*target,icp.transformation_,1.0);
                ROS_INFO("%s:matched point size is: %d \n","icp_corner",icp_result_corner_.inlier_p_cnt_);
            } else if (regis_type == "surf_icp")
            {
                icp_surf.inlier_p_cnt_ = getInlinerIcpPointCount(*source,*target,icp.transformation_,1.0);
                ROS_INFO("%s:matched point size is: %d \n","icp_surf",icp_result_surf_.inlier_p_cnt_);
            }
            break;
        }
        default:
            std::cout << "The insert method is not choose the desred one !!" <<std::endl;
            break;
    }
    return icp_result;
}

size_t OdomEstimationClass::getInlinerIcpPointCount(const open3d::geometry::PointCloud & source ,
    const open3d::geometry::PointCloud & target ,
     const Eigen::Matrix4d & trans_mat , double max_corr_dist)
{
    using namespace open3d;
    /// translation to target
    geometry::PointCloud src_tf = source;
    src_tf.Transform(trans_mat) ;

    geometry::KDTreeFlann kdtree_target(target);

    double max_corr_dist2 = pow(max_corr_dist,2);
    int point_cnt = 0 ;
    std::vector<int> indices(1);
    std::vector<double> distance2(1);
    for(const auto p : src_tf.points_)
    {
        kdtree_target.SearchKNN(p,1,indices,distance2);
        if(distance2[0] <= max_corr_dist2)
        {
            point_cnt++;
        }
        distance2.clear();
        indices.clear();
    }
    return point_cnt ;
}

bool OdomEstimationClass::checkICPResult(const pipelines::registration::RegistrationResult & icp_result_corner ,
    const pipelines::registration::RegistrationResult & icp_result_surf)
{
    icp_result_surf_.T_ = icp_result_surf.transformation_;
    icp_result_surf_.icp_rmse_ = icp_result_surf.inlier_rmse_;
    icp_result_surf_.fitness_ = icp_result_surf.fitness_;
    icp_result_surf_ = calculateIcpQuality(icp_result_surf_);

    icp_result_corner_.T_ = icp_result_corner.transformation_;
    icp_result_corner_.icp_rmse_ = icp_result_corner.inlier_rmse_;
    icp_result_corner_.fitness_ = icp_result_corner.fitness_;
    icp_result_corner_ = calculateIcpQuality(icp_result_corner_);

    if (!icp_result_surf_.valid_ && ! icp_result_corner_.valid_)
    {
        ROS_WARN("corner and surf icp failed, not meet the matching requirements");
        evaluate_metric_icp.accepted = false;
        evaluate_metric_icp.quality = 0.0;
        evaluate_metric_icp.T_final = init_trans_pose_;
        return false;
    }

    if (icp_result_corner_.valid_ && !icp_result_surf_.valid_)
    {
        evaluate_metric_icp.T_final = icp_result_corner_.T_;
        evaluate_metric_icp.quality = (1.0 / (1.0 + icp_result_corner_.icp_rmse_ / 0.3));
        evaluate_metric_icp.accepted = true;
        ROS_INFO("use the corner registration translation for initial guess");
        return true;
    }else if (!icp_result_corner_.valid_ && icp_result_surf_.valid_)
    {
        evaluate_metric_icp.T_final = icp_result_surf_.T_;
        evaluate_metric_icp.quality = (1.0 / (1.0 + icp_result_surf_.icp_rmse_ / 0.3));
        evaluate_metric_icp.accepted = true;
        ROS_INFO("use the surf registration translation for initial guess");
        return true;
    }

    /// both surf icp and corner icp valid
    Eigen::Matrix4d dT = icp_corner.T_.inverse() * icp_surf.T_; /// transformation matrix difference
    Eigen::AngleAxisd aa(Eigen::Matrix3d(dT.block<3,3>(0,0)));
    double dtheta = std::abs(aa.angle()); // rad
    double dt = (dT.block<3,1>(0,3)).norm();  /// translation difference

    //// 计算权重
    double wei_surf = (icp_surf.inlier_p_cnt_ *icp_surf.fitness_)/(0.3 * 0.3 + eps);  // surf weight
    double wei_corner = (icp_corner.inlier_p_cnt_ *icp_corner.fitness_)/(0.3 * 0.3 + eps); // corner weight 
    /// 用于计算组合的误差
    auto s_norm = std::sqrt(
        (wei_surf * std::pow(icp_surf.icp_rmse_ / 0.3, 2) + wei_corner * std::pow(icp_corner.icp_rmse_ / 0.3, 2)) /(wei_surf + wei_corner + eps));

    double q_res = 1.0 / (1.0 + s_norm);
    double q_fit = (wei_surf * icp_surf.fitness_ + wei_corner * icp_corner.fitness_) / (wei_surf + wei_corner + eps); /// fitness weight
    double Q = std::pow(q_res, 1.0) * std::pow(q_fit, 1.0);

    const double thetamax = 5.0 * M_PI / 180.0, tmax = 0.50; ///euler angle and translation difference threshold

    auto logmap = [&](const Eigen::Matrix4d& A) -> Eigen::Matrix<double,6,1> {
        /// 反对称矩阵
        auto skew = [](const Eigen::Vector3d& v) {
            Eigen::Matrix3d m;
            m <<     0.0, -v.z(),  v.y(),
                    v.z(),    0.0, -v.x(),
                    -v.y(),  v.x(),   0.0;
            return m;
        };
        auto vee = [](const Eigen::Matrix3d& M) {
            return Eigen::Vector3d(M(2,1), M(0,2), M(1,0));
        };
        Eigen::Matrix3d R = A.block<3,3>(0,0);
        Eigen::Vector3d t = A.block<3,1>(0,3);
        double cos_theta = std::max(-1.0, std::min(1.0, (R.trace() - 1.0) * 0.5));
        double theta = std::acos(cos_theta);
        Eigen::Vector3d omega;
        Eigen::Matrix3d lnR;
        if (theta < 1e-8) {
            lnR = 0.5 * (R - R.transpose());
            omega = vee(lnR);
            Eigen::Matrix3d Ahat = skew(omega);
            Eigen::Matrix3d V_inv = Eigen::Matrix3d::Identity() - 0.5 * Ahat + (1.0/12.0) * (Ahat * Ahat);
            Eigen::Vector3d rho = V_inv * t;
            Eigen::Matrix<double,6,1> xi;
            xi.head<3>() = omega;
            xi.tail<3>() = rho;
            return xi;
        } else {
            lnR = (theta / (2.0 * std::sin(theta))) * (R - R.transpose());
            omega = vee(lnR);
            Eigen::Matrix3d Ahat = skew(omega);
            double theta2 = theta * theta;
            Eigen::Matrix3d V = Eigen::Matrix3d::Identity()
                                + ((1.0 - std::cos(theta)) / theta2) * Ahat
                                + ((theta - std::sin(theta)) / (theta2 * theta)) * (Ahat * Ahat);
            Eigen::Vector3d rho = V.ldlt().solve(t);
            Eigen::Matrix<double,6,1> xi;
            xi.head<3>() = omega;
            xi.tail<3>() = rho;
            return xi;
        }
    };
    auto expmap = [&](const Eigen::Matrix<double,6,1>& xi) -> Eigen::Matrix4d {
        auto skew = [](const Eigen::Vector3d& v) {
            Eigen::Matrix3d m;
            m <<     0.0, -v.z(),  v.y(),
                    v.z(),    0.0, -v.x(),
                    -v.y(),  v.x(),   0.0;
            return m;
        };
        Eigen::Vector3d omega = xi.head<3>();
        Eigen::Vector3d rho = xi.tail<3>();
        double theta = omega.norm();
        Eigen::Matrix3d Ahat = skew(omega);
        Eigen::Matrix3d R;
        Eigen::Matrix3d V;
        ///小角度近似
        if (theta < 1e-8) {
            // series expansion
            Eigen::Matrix3d Ahat2 = Ahat * Ahat;
            R = Eigen::Matrix3d::Identity() + Ahat + 0.5 * Ahat2;
            V = Eigen::Matrix3d::Identity() + 0.5 * Ahat + (1.0/6.0) * Ahat2;
        } else {
            double s = std::sin(theta);
            double c = std::cos(theta);
            double theta2 = theta * theta;
            Eigen::Matrix3d Ahat2 = Ahat * Ahat;
            ///Rodrigues 公式
            R = Eigen::Matrix3d::Identity() + (s/theta) * Ahat + ((1.0 - c)/theta2) * Ahat2;
            V = Eigen::Matrix3d::Identity()
                + ((1.0 - c)/theta2) * Ahat
                + ((theta - s)/(theta2 * theta)) * Ahat2;
        }
        Eigen::Vector3d t = V * rho;
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = R;
        T.block<3,1>(0,3) = t;
        return T;
    };

    if(dtheta > thetamax || dt > tmax)
    {
        double spn = icp_surf.icp_rmse_ / 0.3;
        double scn = icp_corner.icp_rmse_ / 0.3;
        bool pickSurf = (spn < scn) || (std::abs(spn - scn) < 0.05 && wei_surf >= wei_corner);
        Eigen::Matrix4d T_best = pickSurf ? icp_surf.T_ : icp_corner.T_;
        const double low_thres_Q = 0.4, high_thres_Q = 0.8;
        auto clamp = [&](double Q)
        {
            return (Q < high_thres_Q)?((Q>low_thres_Q)?Q:low_thres_Q):high_thres_Q;
        };
        double trust = clamp(Q);
        Eigen::Matrix4d T_coarse = init_trans_pose_;
        Eigen::Matrix<double,6,1> xi = logmap(T_coarse.inverse() * T_best);
        Eigen::Matrix4d T_final = T_coarse * expmap(trust * xi);
        evaluate_metric_icp.T_final = T_final;
        evaluate_metric_icp.quality = Q;
        evaluate_metric_icp.accepted = true;
        ROS_INFO("surf translation is not equal to corner translation  !! using fuse translation !!");
        return true;
    }
    ///将估计位姿与当前位姿进行融合
    Eigen::Matrix<double,6,1> xi_p = logmap(init_trans_pose_.inverse() * icp_surf.T_);///对数映射
    Eigen::Matrix<double,6,1> xi_c = logmap(init_trans_pose_.inverse() * icp_corner.T_);
    Eigen::Matrix<double,6,1> xi = (wei_surf * xi_p + wei_corner * xi_c) / (wei_surf + wei_corner + eps);
    Eigen::Matrix4d T_final = init_trans_pose_ * expmap(xi); /// 指数映射
    evaluate_metric_icp.T_final = T_final;
    evaluate_metric_icp.quality = Q;
    evaluate_metric_icp.accepted = true;
    ROS_INFO(" using fuse translation !!");
    return true;
}

void OdomEstimationClass::calculateIcpQuality(IcpResult & icp_result_in)
{
    const double rmse_max = 0.2 ;
    const double fit_min = 0.8 ;
    auto metric_FixedQuality =[&](IcpResult & icp_result_in)
    {
        return (icp_result_in.fitness_ >= fit_min) &&
                ((icp_result_in.icp_rmse_ > 0.0 ) && (icp_result_in.icp_rmse_ < rmse_max));
    };
    icp_result_in.valid_ = metric_FixedQuality(icp_result_in);

}

bool OdomEstimationClass::SaveKeyframeRadius(Eigen::Isometry3d & curr_odom ,Eigen::Isometry3d & last_odom)
{
    if(keyframe_measurements_store.empty())
    {
        return true;
    }

    Pose6D delta_pose = getTransformation(curr_odom, last_odom);
    double delta_drz = delta_pose.drz;
    if(delta_pose.drz > M_PI) delta_drz = delta_pose.drz - 2 * M_PI;
    else if(delta_pose.drz < -M_PI) delta_drz = delta_pose.drz + 2 * M_PI;

    bool reach_thres = false;
    double delta_translation = sqrt(pow(delta_pose.dtx,2) + pow(delta_pose.dty,2) + pow(delta_pose.dtz,2));
    keyframe_trans_accumulate += delta_translation;
    if(abs(delta_tf.roll) > keyframe_rot_thres ||
       abs(delta_tf.pitch) > keyframe_rot_thres ||
       abs(delta_yaw) > keyframe_rot_thres ||
       keyframe_trans_accumulate > keyframe_trans_thres)
    {
        reach_thres = true;
        keyframe_trans_accumulate = 0;
    }

    return reach_thres ;


}
Pose6D OdomEstimationClass::getTransformation(Eigen::Isometry3d & curr_odom ,Eigen::Isometry3d & last_odom)
{
    Eigen::Vector3d  curr_euler_zyx = curr_odom.rotation().eulerAngles(2,1,0);
    Eigen::Vector3d curr_translation = curr_odom.translation();
    Pose6D curr_odom_pose = {curr_translation[0],curr_translation[1],curr_translation[2],
                             curr_euler_zyx[2],curr_euler_zyx[1],curr_euler_zyx[0]};

    Eigen::Vector3d prev_euler_zyx = last_odom.rotation().eulerAngles(2,1,0);
    Eigen::Vector3d prev_translation = last_odom.translation();
    Pose6D prev_odom_pose = {prev_translation[0],prev_translation[1],prev_translation[2],
                             prev_euler_zyx[2],prev_euler_zyx[1],prev_euler_zyx[0]};

    Eigen::Affine3f  SE3_pose_prev = pcl::getTransformation(prev_odom_pose.x,prev_odom_pose.y,prev_odom_pose.z,
                                                            prev_odom_pose.roll , prev_odom_pose.pitch,prev_odom_pose.yaw);

    Eigen::Affine3f  SE3_pose_curr = pcl::getTransformation(curr_odom_pose.x,curr_odom_pose.y,curr_odom_pose.z,
                                                            curr_odom_pose.roll , curr_odom_pose.pitch,curr_odom_pose.yaw);

    Eigen::Matrix4f  SE3_delta_1 = SE3_pose_prev.matrix().inverse() * SE3_pose_curr.matrix();
    Eigen::Affine3f SE3_delta_2;
    SE3_delta_2.matrix() = SE3_delta_1;///仿射变换
    float dtx, dty, dtz, drx, dry, drz;
    pcl::getTranslationAndEulerAngles(SE3_delta_2, dtx, dty, dtz, drx, dry, drz);

    return Pose6D(double(abs(dtx)), double(abs(dty)), double(abs(dtz)),
                  double(abs(drx)), double(abs(dry)), double(abs(drz)));
}
