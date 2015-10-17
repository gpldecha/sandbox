#include "machine_learning/node/topic_publisher.h"

namespace ml{

Topic_publisher::Topic_publisher(ros::NodeHandle& node,const std::string& topic_name){

    pub_data    = node.advertise<std_msgs::Float64MultiArray>(topic_name + "_data", 100);
    pub_label   = node.advertise<std_msgs::String>(topic_name + "_label", 100);
    pub_winner  = node.advertise<std_msgs::Int8>(topic_name + "_winner",100);


}

void  Topic_publisher::initialise(const std::vector<std::string> &labels){
    data_msg.data.resize(labels.size());

    label_msg.data = "";

    for(std::size_t i = 0; i < labels.size()-1;i++){
        label_msg.data =  label_msg.data  + labels[i] + "|";
    }
    label_msg.data = label_msg.data + labels[labels.size()-1];

}

void Topic_publisher::update(const arma::colvec& data){

    assert(data.n_elem == data_msg.data.size());

    for(std::size_t i = 0; i < data.n_elem;i++){
        data_msg.data[i] = data(i);
    }

}

void Topic_publisher::update(int label){
    winner_msg.data = label;
}

void Topic_publisher::publish(){

    pub_data.publish(data_msg);
    pub_label.publish(label_msg);
    pub_winner.publish(winner_msg);

}


}
