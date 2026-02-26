#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Dense>
#include <cmath>
#include <string>

class EkfDiffImu : public rclcpp::Node {
public:
  EkfDiffImu() : Node("ekf_diff_imu") {
    // Parameters
    wheel_odom_topic_ = declare_parameter<std::string>("wheel_odom_topic", "/wheel_encoder/odom");
    odom_topic_       = declare_parameter<std::string>("odom_topic", "/ekf_diff_imu/odom");
    imu_topic_        = declare_parameter<std::string>("imu_topic", "/zed/zed_node/imu/data_raw");

    sigma_v_     = declare_parameter<double>("sigma_v", 0.10);
    sigma_omega_ = declare_parameter<double>("sigma_omega", 1e-8);
    sigma_omega_ = std::sqrt(sigma_omega_);

    x_.setZero();
    P_.setIdentity();
    P_ *= 0.1;

    wheel_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      wheel_odom_topic_, rclcpp::SensorDataQoS(),
      std::bind(&EkfDiffImu::wheelOdomCb, this, std::placeholders::_1));

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(),
      std::bind(&EkfDiffImu::imuCb, this, std::placeholders::_1));

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  }

private:
  double sigma_v_, sigma_omega_;
  std::string wheel_odom_topic_, odom_topic_, imu_topic_;

  Eigen::Vector3d x_;   // [x, y, theta]
  Eigen::Matrix3d P_;
  rclcpp::Time last_predict_time_{0,0,RCL_ROS_TIME};
  double last_v_{0.0}, last_omega_{0.0};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  static double wrapToPi(double a) {
    while (a <= -M_PI) a += 2.0 * M_PI;
    while (a >   M_PI) a -= 2.0 * M_PI;
    return a;
  }

  void predict(double v, double omega, double dt) {
    const double th = x_(2);
    const double c = std::cos(th);
    const double s = std::sin(th);

    x_(0) += v * c * dt;
    x_(1) += v * s * dt;
    x_(2) = wrapToPi(th + omega * dt);

    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0,2) = -v * s * dt;
    F(1,2) =  v * c * dt;

    Eigen::Matrix<double,3,2> G;
    G << c*dt, 0.0,
         s*dt, 0.0,
         0.0,  dt;

    Eigen::Matrix2d Qu = Eigen::Matrix2d::Zero();
    Qu(0,0) = sigma_v_ * sigma_v_;
    Qu(1,1) = sigma_omega_ * sigma_omega_;

    Eigen::Matrix3d Q = G * Qu * G.transpose();
    P_ = F * P_ * F.transpose() + Q;
  }

  void updateYaw(double yaw_meas, double var_yaw) {
    Eigen::RowVector3d H;
    H << 0.0, 0.0, 1.0;

    double z_pred = x_(2);
    double y = wrapToPi(yaw_meas - z_pred);

    double S = (H * P_ * H.transpose())(0,0) + var_yaw;
    Eigen::Vector3d K = P_ * H.transpose() / S;

    x_ += K * y;
    x_(2) = wrapToPi(x_(2));

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    P_ = (I - K * H) * P_;
  }

  void publishOdom(const rclcpp::Time& stamp) {
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "odom";
    odom.child_frame_id  = "base_link";

    odom.pose.pose.position.x = x_(0);
    odom.pose.pose.position.y = x_(1);
    odom.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, x_(2));
    q.normalize();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    for (int i=0; i<36; i++) odom.pose.covariance[i] = 0.0;
    odom.pose.covariance[0]  = P_(0,0);
    odom.pose.covariance[7]  = P_(1,1);
    odom.pose.covariance[35] = P_(2,2);

    odom.twist.twist.linear.x  = last_v_;
    odom.twist.twist.angular.z = last_omega_;
    odom_pub_->publish(odom);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = "odom";
    tf_msg.child_frame_id = "base_link";
    tf_msg.transform.translation.x = x_(0);
    tf_msg.transform.translation.y = x_(1);
    tf_msg.transform.translation.z = 0.0;
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(tf_msg);
  }

  void wheelOdomCb(const nav_msgs::msg::Odometry::SharedPtr msg) {
    double v = msg->twist.twist.linear.x;
    double omega = msg->twist.twist.angular.z;
    rclcpp::Time t = msg->header.stamp;

    if (!last_predict_time_.nanoseconds()) {
      last_predict_time_ = t;
    }

    double dt = (t - last_predict_time_).seconds();
    if (dt <= 1e-4) dt = 1e-4;

    predict(v, omega, dt);
    last_v_ = v;
    last_omega_ = omega;

    publishOdom(t);
    last_predict_time_ = t;
  }

  void imuCb(const sensor_msgs::msg::Imu::SharedPtr msg) {
    const auto &q = msg->orientation;
    tf2::Quaternion qt(q.x, q.y, q.z, q.w);
    qt.normalize();
    double roll, pitch, yaw;
    tf2::Matrix3x3(qt).getRPY(roll, pitch, yaw);

    double var_yaw = 1e-8;
    if (msg->orientation_covariance[8] >= 0.0) {
      var_yaw = std::max(msg->orientation_covariance[8], 1e-8);
    }

    updateYaw(yaw, var_yaw);
    publishOdom(msg->header.stamp);
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EkfDiffImu>());
  rclcpp::shutdown();
  return 0;
}
