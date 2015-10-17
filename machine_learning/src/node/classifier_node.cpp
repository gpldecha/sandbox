#include <optitrack_rviz/input.h>
#include <optitrack_rviz/listener.h>

#include <ros/ros.h>

#include <map>
#include <string>
#include <memory>

#include "machine_learning/node/classifier_interface.h"
#include "machine_learning/node/topic_publisher.h"


int main(int argc,char** argv){


    std::map<std::string,std::string> input;
    input["-class_type"]       = "gmm";
    input["-parameters"]       = "";
    input["-topic_listen"]     = "";
    input["-topic_pub"]        = "/classifier";
    input["-rate"]             = "100";


    if(!opti_rviz::Input::process_input(argc,argv,input)){
         ROS_ERROR("failed to load input");
         return 0;
     }
     opti_rviz::Input::print_input_options(input);

     ros::init(argc, argv, "machine_learning_classifier");
     ros::NodeHandle node;
     ros::Rate rate(boost::lexical_cast<float>(input["-rate"]));

     ml::Classifier_interface   classifier_interface(input["-class_type"],input["-parameters"]);

     assert(classifier_interface.ptr_classifier != NULL);

     std::size_t num_classes                =  classifier_interface.ptr_classifier->get_num_classes();
     std::size_t data_size                  =  classifier_interface.ptr_classifier->get_data_size();
     std::vector<std::string>& class_labels =  classifier_interface.ptr_classifier->get_class_labels();

  //   ml::Topic_listener         topic_listener(node,input["-topic_listen"],data_size);
     ml::Topic_publisher        topic_publiser(node,input["-topic_pub"]);
                                topic_publiser.initialise(class_labels);

     ROS_INFO("=== classifier node ready! ===");
     ROS_INFO(" data size:   %d",data_size);
     ROS_INFO(" num classes: %d",num_classes);


     ros::spinOnce();

     int label = -1;

     while(node.ok()){

         //label = classifier_interface.ptr_classifier->predict(topic_listener.data);

         topic_publiser.update(classifier_interface.ptr_classifier->get_probabilities());
         topic_publiser.update(label);

         topic_publiser.publish();

         ros::spinOnce();
         rate.sleep();
     }

    return 0;
}
