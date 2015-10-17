#include <statistics/clustering.h>
#include <statistics/distributions/gmm.h>
#include <statistics/meanshift.h>

#include <visualise/vis_point_cloud.h>
#include <visualise/vis_gmm.h>
#include <visualise/vis_point_cloud.h>
#include <visualise/colormap.h>
#include <visualise/vis_points.h>

int main(int argc, char** argv){

    ros::init(argc, argv, "meanshift_test");
    ros::NodeHandle node;

    std::size_t K = 5;

    arma::vec pi = arma::randu<arma::vec>(3);
              pi = arma::normalise(pi);
    std::vector<arma::vec> Mu(K);
    std::vector<arma::mat> Sigma(K);


    float a = -4;
    float b =  4;

    for(std::size_t k = 0; k < K;k++){
        Mu[k] = (b - a) * arma::randu<arma::vec>(3) + a;
        Sigma[k].zeros(3,3);
        arma::vec tmp = 0.2 * arma::randu<arma::vec>(3) + 0.1;
        Sigma[k](0,0) =tmp(0);
        Sigma[k](1,1) =tmp(1);
        Sigma[k](2,2) =tmp(2);
    }


      GMM gmm;
      gmm.setParam(pi,Mu,Sigma);
      std::size_t nb_samples = 10;
      arma::colvec weights;
      arma::mat X(nb_samples,3);
      gmm.sample(X);
      arma::vec L(nb_samples);
      gmm.P(X,L);

      L = L / arma::max(L);

      unsigned char                           rgb[3];
      std::vector<std::array<float,3> >       colors(nb_samples);

      for(std::size_t i = 0 ; i < L.n_elem;i++){
          ColorMap::jetColorMap(rgb,L(i),0,1);
          colors[i][0]    = ((float)rgb[0])/255;
          colors[i][1]    = ((float)rgb[1])/255;
          colors[i][2]    = ((float)rgb[2])/255;
      }

      mean_shift::MeanShift_Parameters mean_shift_parameters;
      mean_shift_parameters.bandwidth = 1.0/(0.1 * 0.1);
      mean_shift::MeanShift meanshift(X,mean_shift_parameters);

      arma::mat init_centroids(20,3);
      gmm.sample(init_centroids);

      std::cout<< "==== run meanshift === " << std::endl;

      meanshift.set_initial_center_guess(init_centroids);
      meanshift.update();

      meanshift.centroids.print("centroids");
      meanshift.merge_centroids(0.001);
      meanshift.modes.print("modes");

      std::cout<< "==== run kmeans clustering + gmm ====" << std::endl;

      Clustering clustering;
      GMM        gmm2;

      arma::mat init_centers = meanshift.modes.st();
      clustering.setInitCentroids(init_centers);

      X  = X.st();
      clustering.kmeans(X);
      X  = X.st();
      clustering.mixture_model(gmm2,X,L);

      opti_rviz::Vis_gmm vis_gmm(node,"gmm");
      vis_gmm.initialise("world",gmm2.gmm.Weights(),gmm2.gmm.Means(),gmm2.gmm.Covariances());
//      vis_gmm.update(gmm2.gmm.Weights(),gmm2.gmm.Means(),gmm2.gmm.Covariances());


      arma::mat cov_x = arma::cov(X);
      arma::colvec mu = arma::mean(X,0).st();

      std::cout<< "cov_x: " << cov_x.n_rows << " x " << cov_x.n_cols << std::endl;
      mu.print("mean");

      opti_rviz::Vis_points points(node,"centroids_start");
      points.scale = 0.1;
      points.r     = 0;
      points.g     = 1;
      arma::fmat c = arma::conv_to<arma::fmat>::from(init_centroids);
      points.initialise("world",c);

      opti_rviz::Vis_points points2(node,"centroids_end");
      points2.scale = 0.1;
      points2.r     = 1;
      points2.g     = 0;
      arma::fmat c_end = arma::conv_to<arma::fmat>::from(meanshift.centroids);
      points2.initialise("world",c_end);

      opti_rviz::Vis_point_cloud vis_point(node,"samples");
      vis_point.set_display_type(opti_rviz::Vis_point_cloud::DEFAULT);
      vis_point.initialise("world",X);


    ros::Rate rate(100);


    while(node.ok()){


        vis_point.update(X,colors,weights);
        vis_point.publish();

        points.update(c);
        points.publish();

        vis_gmm.publish();

        //meanshift.one_step_update();
        c_end = arma::conv_to<arma::fmat>::from(meanshift.centroids);

        points2.update(c_end);
        points2.publish();

        ros::spinOnce();
        rate.sleep();
    }



}
