// Author of FLOAM: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#include "odomEstimationClass.h"
int count_surf = 0;
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
    pcl::getMinMax3D<pcl::PointXYZRGB> (*input, min_p, max_p);

    min_b_[0] = static_cast<int> (floor (min_p[0] / dsleaf));
	max_b_[0] = static_cast<int> (floor (max_p[0] / dsleaf));
	min_b_[1] = static_cast<int> (floor (min_p[1] / dsleaf));
	max_b_[1] = static_cast<int> (floor (max_p[1] / dsleaf));
	min_b_[2] = static_cast<int> (floor (min_p[2] / dsleaf));
	max_b_[2] = static_cast<int> (floor (max_p[2] / dsleaf));

    div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones ();
	div_b_[3] = 0;
	divb_mul_ = Eigen::Vector4i (1, div_b_[0], div_b_[0] * div_b_[1], 0);

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

void OdomEstimationClass::init(lidar::Lidar lidar_param, double map_resolution, double _k, double _theta, double _king){

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
}

void OdomEstimationClass::initMapWithPoints(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_in){
    // 添加特征点到Map
    *laserCloudCornerMap += *edge_in;
    *laserCloudSurfMap += *surf_in;
    optimization_count=12;
}


void OdomEstimationClass::updatePointsToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& surf_in){

    // 为什么需要把优化次数慢慢降下来？？
    if(optimization_count>2)
        optimization_count--;

    // 利用前一次位姿变换来预测当前位姿变换 odom * last_odom.inverse 相当于odom相对于last_odom的增量
    Eigen::Isometry3d odom_prediction = odom * (last_odom.inverse() * odom);
    last_odom = odom;
    odom = odom_prediction;

    // 储存当前点云的位姿
    q_w_curr = Eigen::Quaterniond(odom.rotation());
    t_w_curr = odom.translation();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampledEdgeCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampledSurfCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    // 下采样两个特征点云
    downSamplingToMap(edge_in,downsampledEdgeCloud,surf_in,downsampledSurfCloud);
    //ROS_WARN("point nyum%d,%d",(int)downsampledEdgeCloud->points.size(), (int)downsampledSurfCloud->points.size());
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

        TicToc opt_time ;
        // 优化optimization_count轮（每次会对parameter在前一次基础上进行优化）
        for (int iterCount = 0; iterCount < optimization_count; iterCount++){
            {
                /// surf optimization 
                {
                    ceres::LossFunction *loss_function_s = new ceres::HuberLoss(0.1);
                    ceres::Problem::Options surf_opt;
                    ceres::Problem surf_pbm(surf_opt);
                    surf_pbm.AddParameterBlock(paramEuler, 6);
                    
                    // 添加面特征项
                    addSurfCostFactor(downsampledSurfCloud,laserCloudSurfMap,surf_pbm,loss_function_s);

                    /// 筛选最优top K
                    std::vector<size_t> surf_idx = selectTopNSurf(2000,surf_corrs_vec);


                    /// top k 加入面特征residual 
                    for(size_t k_s = 0 ; k_s < surf_idx.size() ; k_s ++)
                    {
                        size_t i = surf_idx[k_s];
                        const auto& sc = surf_corrs_vec[i];
                        ceres::CostFunction *cost_function_s = new SurfNormAnalyticCostFunction(sc.pc_, sc.norm, sc.neg, sc.wei_s);
                        surf_pbm.AddResidualBlock(cost_function_s, loss_function_s, paramEuler);
                    }
                    /// 面特征优化求解
                    ceres::Solver::Options options_s;
                    options_s.max_num_iterations = 4;
                    options_s.minimizer_progress_to_stdout = false;
                    options_s.check_gradients = false;
                    options_s.gradient_check_relative_precision = 1e-4;
                    options_s.linear_solver_type = ceres::DENSE_QR;
                    ceres::Solver::Summary summary_s;
                    ceres::Solve(options_s, &surf_pbm, &summary_s);
                    updatePose();
                }
                /// corner optimization 
                // {
                //     addEdgeCostFactor(downsampledEdgeCloud,laserCloudCornerMap,problem_joint,loss_function);
                // }
                
                // 

                // /// 筛选top k residual seed 
               
                // std::vector<size_t> corner_idx = selectTopNCorner(5000,cor_corrs_vec);

                // /// add surf residual (weight)


                // /// add corner residual (weight)
                // for(size_t k_c = 0 ; k_c < corner_idx.size() ; k_c ++)
                // {
                //     size_t i = corner_idx[k_c];
                //     const auto& cc = cor_corrs_vec[i];
                //     ROS_INFO("直线的权重 : %f",cc.wei_c);
                //     ceres::CostFunction *cost = new EdgeAnalyticCostFunction(cc.pc_, cc.pa_, cc.pb_);
                //     problem_joint.AddResidualBlock(cost, loss_function, paramEuler);
                // }
                

                // // 优化参数设置
                // ceres::Solver::Options options;
                // options.linear_solver_type = ceres::DENSE_QR;
                // options.max_num_iterations = 4;
                // options.minimizer_progress_to_stdout = false;
                // options.check_gradients = false;
                // options.gradient_check_relative_precision = 1e-4;
                // ceres::Solver::Summary summary;
                // ceres::Solve(options, &problem_joint, &summary);

                // updatePose();
            }
        }
        double opt_duration = opt_time.toc();
        ROS_INFO("optimization time duration is %f ms", opt_duration);

    }else{
        printf("not enough points in map to associate, map error");
    }
    // 更新odom坐标，为下一次prediction作准备
    odom = Eigen::Isometry3d::Identity();
    odom.linear() = q_w_curr.toRotationMatrix();
    odom.translation() = t_w_curr;

    // 添加特征点到map中
    addPointsToMap(downsampledEdgeCloud,downsampledSurfCloud);

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

    cor_corrs_vec.clear();
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
                    map_in->points[pointSearchInd[j]].g = std::min(255, map_in->points[pointSearchInd[j]].g + 1);
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
                // ceres::CostFunction *cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b);  
                // problem.AddResidualBlock(cost_function, loss_function, paramEuler);
                corner_num++;   
                
                /// calculate the edge residual weight 
                {

                }
                Eigen::Vector3d p_c(pc_in->points[i].x , pc_in->points[i].y ,pc_in->points[i].z);
                Eigen::Vector3d p_w = R_init * p_c + t_init;
                Eigen::Vector3d n_u= (p_w - point_a).cross(p_w - point_b);
                Eigen::Vector3d d_e = point_a - point_b ;

                /// calculate the line weight 
                double l1 = saes.eigenvalues()[0] ;
                double l2 = saes.eigenvalues()[1] ;
                double l3 = saes.eigenvalues()[2];

                /// 沿线跨度
                double maxp = -1e9, minp = 1e9;
                double max_c = -1e9 , min_c = 1e9;
                for (int j = 0; j < 5; ++j){
                    double proj = (nearCorners[j] - center).dot(unit_direction);
                    if (proj > maxp) 
                    {
                        maxp = proj;
                        max_c = (nearCorners[j] - center).norm();
                    }
                    
                    if (proj < minp)
                    {
                        minp = proj;
                        min_c = (nearCorners[j] - center).norm();
                    }
                }
                double span = std::max(0.0,maxp - minp);
                double max_c_min = max_c + min_c;

                double w_geom_edge = std::min(1.0, span / max_c_min);

                double res = n_u.norm() / std::max(1e-9,d_e.norm());
                ROS_INFO("corner residual is %f , line weight is %f",res,w_geom_edge);
                /// save residual corner 
                cor_corrs_vec.push_back(Corner_Corr{p_c,point_a,point_b,w_geom_edge});
            }                           
        }
    }
    ROS_INFO("corner residual number is %d ",corner_num);
    if(corner_num<20){
        printf("not enough correct points");
    }
}

void OdomEstimationClass::addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pc_in, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& map_in, ceres::Problem& problem, ceres::LossFunction *loss_function){
    int surf_num=0;
    surf_corrs_vec.clear();
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {
        pcl::PointXYZRGB point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);  // r t
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
            std::vector<double> res_vec ;
            double res_mean = 0.0 ;

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
                /// insert the residual 
                res_vec.push_back(fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
                         norm(1) * map_in->points[pointSearchInd[j]].y +
                         norm(2) * map_in->points[pointSearchInd[j]].z + negative_OA_dot_norm));
                res_mean += fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
                         norm(1) * map_in->points[pointSearchInd[j]].y +
                         norm(2) * map_in->points[pointSearchInd[j]].z + negative_OA_dot_norm);
            }
            /// average error 
            res_mean /= 5.0;
            // ROS_INFO("surf mean residual is %f" ,res_mean);
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
                // ceres::CostFunction *cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, pc_in->points[i].b);    
                // problem.AddResidualBlock(cost_function, loss_function, paramEuler);
                // surf corr
                {
                    std::sort(res_vec.begin(),res_vec.end());
                    double mid = res_vec[2];
                    // ROS_INFO("residual mid is %f" ,mid);
                    for(int i = 0 ; i < res_vec.size() ; i++)
                    {
                        res_vec[i] = std::fabs(res_vec[i] - mid);
                    }
                    std::sort(res_vec.begin() ,res_vec.end());
                    double mad = res_vec[2];

                    /// 几何权重
                    double res_mean_ = 0.1 ,res_mad = 0.05;
                    //// 平面度
                    double w_plan = 1.0 / (1.0 + pow(res_mean /res_mean_ ,2 ));/// < 1
                    double w_stab = 1.0 / (1.0 + std::pow(mad / res_mad ,2)); /// <1
                    double wei_surf = std::min(1.0 , std::max(0.1,w_plan * w_stab));
                    // ROS_INFO("surf planierity is %f , stability is %f ,surf weight is %f" , w_plan,w_stab ,wei_surf);
                    /// save residual corner 
                    surf_corrs_vec.push_back(Surf_Corr{curr_point,norm,negative_OA_dot_norm,pc_in->points[i].b,wei_surf});
                }
                surf_num++;
            }
        }
    }
    if(surf_num<20){
        printf("not enough correct points");
        return ;
    }
}

// 添加特征点到edge和surf地图中
void OdomEstimationClass::addPointsToMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampledEdgeCloud, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& downsampledSurfCloud){

    // 将cloud中经过优化完成的变换矩阵的变换，然后添加到对应的地图中
    for (int i = 0; i < (int)downsampledEdgeCloud->points.size(); i++)
    {
        pcl::PointXYZRGB point_temp;
        pointAssociateToMap(&downsampledEdgeCloud->points[i], &point_temp);
        laserCloudCornerMap->push_back(point_temp); 
    }
    
    for (int i = 0; i < (int)downsampledSurfCloud->points.size(); i++)
    {
        pcl::PointXYZRGB point_temp;
        pointAssociateToMap(&downsampledSurfCloud->points[i], &point_temp);
        laserCloudSurfMap->push_back(point_temp);
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
    // downSizeFilterSurf.setInputCloud(tmpSurf);
    // downSizeFilterSurf.filter(*laserCloudSurfMap);
    // downSizeFilterEdge.setInputCloud(tmpCorner);
    // downSizeFilterEdge.filter(*laserCloudCornerMap);    
    
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

}

void OdomEstimationClass::getMap(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& laserCloudMap){
    
    *laserCloudMap += *laserCloudSurfMap;
    *laserCloudMap += *laserCloudCornerMap;
}

OdomEstimationClass::OdomEstimationClass(){

}

double OdomEstimationClass::median(std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double m = v[mid];
    if ((n & 1) == 0) {
        double m2 = *std::max_element(v.begin(), v.begin() + mid);
        m = 0.5 * (m + m2);
    }
    return m;
}

double OdomEstimationClass::RobustMADEstimation(const std::vector<double> res_vec_in)
{
    /// calculate the median 
    auto median = [&](std::vector<double>& v)->double{
    if (v.empty()) return 0.0;
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double m = v[mid];
    if ((n & 1) == 0) {
            double m2 = *std::max_element(v.begin(), v.begin() + mid);
            m = 0.5 * (m + m2);
        }
    return m;
    };
    if (res_vec_in.size() < 6) 
        return 0.0;
    std::vector<double> r = res_vec_in;
    double r_med = median(r);
    std::vector<double> abs_dev;
    abs_dev.reserve(res_vec_in.size());
    for (double x : res_vec_in) 
    {
        abs_dev.push_back(fabs(x - r_med));
    }
    double mad = median(abs_dev);
    return 1.4826 * mad;
};

std::vector<size_t> OdomEstimationClass::selectTopNCorner(size_t N, const std::vector<Corner_Corr> & res_cor)
{
    std::vector<size_t> idx(res_cor.size());
    std::iota(idx.begin(), idx.end(), 0);

    auto key = [&](size_t i){ return res_cor[i].wei_c; };
    if (idx.size() > N){
        std::nth_element(idx.begin(), idx.begin()+N, idx.end(), [&](size_t a, size_t b){ return key(a) > key(b); });
        // ROS_INFO("select top 2000 residual for factor graph optimization");
        idx.resize(N);
    }
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return key(a) > key(b); });
    return idx;
}

std::vector<size_t> OdomEstimationClass::selectTopNSurf(size_t N, const std::vector<Surf_Corr> & surf_corr_vec)
{
    std::vector<size_t> idx(surf_corr_vec.size());
    /// set zero 
    std::iota(idx.begin(), idx.end(), 0);

    auto key = [&](size_t i){ return surf_corrs_vec[i].wei_s; };
    if (idx.size() > N){
        std::nth_element(idx.begin(), idx.begin()+N, idx.end(), [
            &](size_t a, size_t b){ return key(a) > key(b); });
        ROS_INFO("select top %d residual for factor graph optimization",N);
        idx.resize(N);
    }
    /// from large to small 
    std::sort(idx.begin(), idx.end(),
     [&](size_t a, size_t b){ return key(a) > key(b);});
    return idx;
}

Eigen::Isometry3d OdomEstimationClass::ParamToIso(const double  p[6])
{
    double rx = p[0], ry = p[1], rz = p[2];
    double tx = p[3], ty = p[4], tz = p[5];
    Eigen::Matrix3d R = (
        Eigen::AngleAxisd(rz, Eigen::Vector3d::UnitZ()) * 
        Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitY()) * 
        Eigen::AngleAxisd(rx, Eigen::Vector3d::UnitX())
    ).toRotationMatrix();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = R;
    T.translation() = Eigen::Vector3d(tx,ty,tz);
    return T;
}
