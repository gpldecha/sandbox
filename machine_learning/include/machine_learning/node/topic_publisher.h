#ifndef TOPIC_PUBLISHER_H_
#define TOPIC_PUBLISHER_H_

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <std_msgs/Int8.h>

#include <armadillo>

namespace ml {

class Topic_publisher{

public:

    Topic_publisher(ros::NodeHandle& node,const std::string& topic_name);

    void initialise(const std::vector<std::string>& labels);

    void update(const arma::colvec& data);

    void update(int label);

    void publish();

private:

    ros::Publisher              pub_data;
    ros::Publisher              pub_label;
    ros::Publisher              pub_winner;

    std_msgs::Float64MultiArray data_msg;
    std_msgs::String            label_msg;
    std_msgs::Int8              winner_msg;


};

}

#endif
