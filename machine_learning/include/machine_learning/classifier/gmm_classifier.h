#ifndef GMM_CLASSIFIER_H_
#define GMM_CLASSIFIER_H_

#include "machine_learning/classifier/classifier.h"
#include <statistics/distributions/gmm.h>
#include <boost/filesystem.hpp>

namespace ml{

class Gmm_classifier: public Classifier {

public:

    Gmm_classifier(std::string path_to_parameters);

    virtual int predict(const arma::colvec &X);

    std::string& get_predicted_class();

    std::vector<GMM>& get_gmms();

private:

    bool load_parameters(const std::string& path_to_parameters);

    bool get_folders_in_dir(const std::string& path_to_parameters,std::vector<boost::filesystem::path>& paths_to_gmms);

public:

    std::vector<std::string>             gmm_names;


private:

    std::vector<GMM>                     gmms;
    std::vector<boost::filesystem::path> parameters_paths;
    arma::uword                          min_index;



};

}


#endif
