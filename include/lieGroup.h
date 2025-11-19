#ifndef _LIE_GROUT_H_
#define _LIE_GROUT_H_
//反对称矩阵
auto hat = [](const Eigen::Vector3d& w)->Eigen::Matrix3d{
            Eigen::Matrix3d W;
            W <<     0, -w.z(),  w.y(),
                    w.z(),     0, -w.x(),
                    -w.y(),  w.x(),     0;
            return W;
        };

// 指数映射
auto RotExpSo3 = [&](const Eigen::Vector3d& w)->Eigen::Matrix3d{
    double theta = w.norm();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (theta < 1e-9) {
        Eigen::Matrix3d W = hat(w);
        return I + W + 0.5 * (W * W);
    }
    Eigen::Matrix3d W = hat(w);
    double A = sin(theta)/theta;
    double B = (1.0 - cos(theta))/(theta*theta);
    return I + A * W + B * (W*W);
};

auto RotLogSo3 = [&](const Eigen::Matrix3d& R)->Eigen::Vector3d{
    double cos_theta = (R.trace() - 1.0) * 0.5;
    cos_theta = std::min(1.0, std::max(-1.0, cos_theta));
    double theta = acos(cos_theta);
    if (theta < 1e-9) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Matrix3d lnR = (theta/(2.0*sin(theta))) * (R - R.transpose());
    return Eigen::Vector3d(lnR(2,1), lnR(0,2), lnR(1,0));
};

//transform
auto IsoExpSe3 = [&](const Eigen::Matrix<double,6,1>& xi)->Eigen::Isometry3d{
    Eigen::Vector3d w = xi.head<3>();
    Eigen::Vector3d v = xi.tail<3>();
    double theta = w.norm();
    Eigen::Matrix3d R = so3Exp(w);
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d W = hat(w);
    Eigen::Matrix3d V;
    if (theta < 1e-9) {
        V = I + 0.5 * W + (1.0/6.0) * (W*W);
    } else {
        double A = sin(theta)/theta;
        double B = (1.0 - cos(theta))/(theta*theta);
        double C = (theta - sin(theta))/(theta*theta*theta);
        V = I + B * W + C * (W*W);
    }
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = R;
    T.translation() = V * v;
    return T;
};
auto IsoLogSe3 = [&](const Eigen::Isometry3d& T)->Eigen::Matrix<double,6,1>{
    Eigen::Matrix3d R = T.rotation();
    Eigen::Vector3d t = T.translation();
    Eigen::Vector3d w = so3Log(R);
    double theta = w.norm();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d W = hat(w);
    Eigen::Matrix3d V;
    if (theta < 1e-9) {
        V = I + 0.5 * W + (1.0/6.0) * (W*W);
    } else {
        double A = sin(theta)/theta;
        double B = (1.0 - cos(theta))/(theta*theta);
        double C = (theta - sin(theta))/(theta*theta*theta);
        V = I + B * W + C * (W*W);
    }
    // 直接求解 V * v = t
    Eigen::Vector3d v = V.ldlt().solve(t);
    Eigen::Matrix<double,6,1> xi;
    xi.head<3>() = w;
    xi.tail<3>() = v;
    return xi;
};
//iso <--> param
auto paramToIso = [&](const double p[6])->Eigen::Isometry3d{
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
};
auto isoToParam = [&](const Eigen::Isometry3d& T, double p_out[6]){
    Eigen::Vector3d euler_zyx = T.rotation().eulerAngles(2,1,0); // [rz, ry, rx]
    p_out[0] = euler_zyx[2];
    p_out[1] = euler_zyx[1];
    p_out[2] = euler_zyx[0];
    p_out[3] = T.translation().x();
    p_out[4] = T.translation().y();
    p_out[5] = T.translation().z();
};


#endif // _LIE_GROUT_H_