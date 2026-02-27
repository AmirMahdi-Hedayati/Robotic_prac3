#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

class PointCloudFrameIdConverter : public rclcpp::Node
{
public:
  PointCloudFrameIdConverter()
  : Node("frame_id_converter_node")
  {
    // Publisher: republish PointCloud2 with corrected frame_id
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/points",
      rclcpp::QoS(rclcpp::SensorDataQoS())   // BestEffort QoS for sensor data
    );

    // Subscriber: listen to Gazebo point cloud
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/gz_lidar/points",
      rclcpp::QoS(rclcpp::SensorDataQoS()),
      std::bind(&PointCloudFrameIdConverter::pointCloudCallback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(),
                "PointCloud FrameIdConverter started. Listening to /gz_lidar/points...");
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    auto new_msg = *msg;                     // Copy original message
    new_msg.header.frame_id = "rplidar_c1";  // Set correct TF frame
    pub_->publish(new_msg);                  // Publish updated point cloud
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudFrameIdConverter>());
  rclcpp::shutdown();
  return 0;
}
