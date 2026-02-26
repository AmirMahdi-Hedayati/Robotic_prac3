#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class MotorCommandNode : public rclcpp::Node
{
public:
    MotorCommandNode() : Node("motor_command_node")
    {
        // Publishers
        left_motor_pub_ = this->create_publisher<std_msgs::msg::Float64>("/left_motor_rpm", 10);
        right_motor_pub_ = this->create_publisher<std_msgs::msg::Float64>("/right_motor_rpm", 10);

        // Subscriber to motor_commands array
        motor_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/motor_commands",
            10,
            std::bind(&MotorCommandNode::motorCallback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "Motor Command Node started.");
    }

private:
    void motorCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
    {
        if (msg->data.size() < 2)
        {
            RCLCPP_WARN(this->get_logger(), "Motor commands array too short!");
            return;
        }

        // Create Float64 messages
        std_msgs::msg::Float64 left_msg;
        std_msgs::msg::Float64 right_msg;

        left_msg.data = msg->data[0];
        right_msg.data = msg->data[1];

        // Publish to individual topics
        left_motor_pub_->publish(left_msg);
        right_motor_pub_->publish(right_msg);
    }

    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_motor_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr right_motor_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr motor_sub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MotorCommandNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
