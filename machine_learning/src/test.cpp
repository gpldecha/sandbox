#include "machine_learning/classifier/gmm_classifier.h"


int main(int argc, char** argv)
{

    std::cout<< " ===== Testing ===== " << std::endl;

    std::string path_to_paramters = "/home/guillaume/MatlabWorkSpace/plug_sensor_model/Gmm_models";
    ml::Gmm_classifier gmm_classifier(path_to_paramters);

    std::vector<GMM>& gmms = gmm_classifier.get_gmms();

  //  int i = 0;
  //  gmms[i].print();
  //  std::cout<< gmm_classifier.gmm_names[i] << std::endl;



    arma::colvec x1 = {{-4.9728,-1.6767,-1.5828}};

    std::cout<< "predict: " << gmm_classifier.predict(x1) << std::endl;
    std::cout<< "class:   " << gmm_classifier.get_predicted_class() << std::endl;
    gmm_classifier.get_probabilities().print("probabilities");


    return 0;
}
