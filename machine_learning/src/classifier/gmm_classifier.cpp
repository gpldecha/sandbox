#include "machine_learning/classifier/gmm_classifier.h"
#include <boost/filesystem.hpp>

namespace ml {

Gmm_classifier::Gmm_classifier(std::string path_to_parameters):
Classifier()
{

    load_parameters(path_to_parameters);

}


int Gmm_classifier::predict(const arma::colvec &X) {
    for(std::size_t i = 0; i < num_classes;i++){
        prob_classes(i) = gmms[i].nlikelihood(X);
    }
    prob_classes.min(min_index);
    return static_cast<int>(min_index);
}

std::string& Gmm_classifier::get_predicted_class(){
    return gmm_names[static_cast<std::size_t>(min_index)];
}

bool Gmm_classifier::load_parameters(const std::string& path_to_parameters){

    std::vector<boost::filesystem::path> paths_to_gmms;

    if(!get_folders_in_dir(path_to_parameters,paths_to_gmms)){
        return false;
    }

    num_classes = paths_to_gmms.size();
    gmms.resize(num_classes);
    prob_classes.resize(num_classes);
    gmm_names.resize(num_classes);
    class_labels.resize(num_classes);

    for(std::size_t i = 0; i < paths_to_gmms.size();i++){
        std::cout << paths_to_gmms[i] << std::endl;
        gmms[i].load(paths_to_gmms[i].string() + "/");
        gmm_names[i]        = paths_to_gmms[i].filename().string();
        class_labels[i]     = paths_to_gmms[i].filename().string();
    }
    if(gmms.size() > 0){
        data_size  = gmms[0].D;
    }

    return true;
}

bool Gmm_classifier::get_folders_in_dir(const std::string& path_to_parameters,
                                        std::vector<boost::filesystem::path>& paths_to_gmm){

    namespace fs = boost::filesystem;
    fs::path someDir(path_to_parameters);
    fs::directory_iterator end_iter;

    if(!fs::exists(someDir) && !fs::is_directory(someDir)){
        std::cerr << "Gmm_classifier::get_folders_in_dir no such directory: " << path_to_parameters << std::endl;
        return false;
    }

    if ( fs::exists(someDir) && fs::is_directory(someDir))
    {
      for( fs::directory_iterator dir_iter(someDir) ; dir_iter != end_iter ; ++dir_iter)
      {
          paths_to_gmm.push_back(dir_iter->path());
      }
    }
    return true;
}

std::vector<GMM>& Gmm_classifier::get_gmms(){
    return gmms;
}


}
