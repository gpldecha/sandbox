#include <statistics/clustering.h>
#include <statistics/distributions/gmm.h>
#include <visualise/vis_point_cloud.h>
#include <visualise/vis_gmm.h>
#include <visualise/vis_point_cloud.h>
#include <visualise/colormap.h>
#include <visualise/vis_points.h>

int main(int argc, char** argv){

    ros::init(argc, argv, "wkmeans_test");
    ros::NodeHandle node;
    ros::Rate rate(100);

    std::size_t K = 10;

    arma::vec pi = arma::randu<arma::vec>(3);
              pi = arma::normalise(pi);
    std::vector<arma::vec> Mu(K);
    std::vector<arma::mat> Sigma(K);


    float a = -2;
    float b =  2;

    for(std::size_t k = 0; k < K;k++){
        Mu[k] = (b - a) * arma::randu<arma::vec>(3) + a;
        Sigma[k].zeros(3,3);
        arma::vec tmp = 0.3 * arma::randu<arma::vec>(3) + 0.2;
        Sigma[k](0,0) =tmp(0);
        Sigma[k](1,1) =tmp(1);
        Sigma[k](2,2) =tmp(2);
    }


      GMM gmm;
      gmm.setParam(pi,Mu,Sigma);
      std::size_t nb_samples = 10000;
      arma::colvec weights;
      arma::mat X(nb_samples,3);
      gmm.sample(X);
      arma::vec L(nb_samples);
      gmm.P(X,L);

      L = L / arma::max(L);

     // L.elem(q1).zeros();

      unsigned char                           rgb[3];
      std::vector<std::array<float,3> >       colors(nb_samples);


      for(std::size_t i = 0 ; i < L.n_elem;i++){
          ColorMap::jetColorMap(rgb,L(i),0,1);
          colors[i][0]    = ((float)rgb[0])/255;
          colors[i][1]    = ((float)rgb[1])/255;
          colors[i][2]    = ((float)rgb[2])/255;
      }


        arma::uvec q1 = arma::find(L > 0.6*arma::max(L));

        std::vector<std::array<float,3> >  colors2(q1.n_elem);
        arma::mat X_W(q1.n_elem,3);
        for(std::size_t i = 0; i < q1.n_elem;i++){
            assert(q1(i) < colors.size());
            assert(q1(i) < X.n_rows);
            colors2[i]  = colors[q1(i)];
            X_W.row(i)  = X.row(q1(i));
        }




      Weighted_Kmeans<double> weighted_kmeans;

      weighted_kmeans.cluster(X_W.st(),L);
      weighted_kmeans.centroids.print("centroids");

      opti_rviz::Vis_points points(node,"centroids");
      points.scale = 0.1;
      points.r = 1;
      points.b = 1;
      arma::fmat c = arma::conv_to<arma::fmat>::from(weighted_kmeans.centroids.st());
      points.initialise("world",c);

      opti_rviz::Vis_point_cloud vis_point(node,"samples");
      vis_point.set_display_type(opti_rviz::Vis_point_cloud::DEFAULT);
      vis_point.initialise("world",X_W);



    opti_rviz::Vis_gmm vis_gmm(node,"gmm");
    vis_gmm.initialise("world",pi,Mu,Sigma);


    while(node.ok()){


        vis_gmm.update(pi,Mu,Sigma);
        vis_gmm.publish();

        vis_point.update(X_W,colors2,weights);
        vis_point.publish();

        points.update(c);
        points.publish();

        ros::spinOnce();
        rate.sleep();
    }



}
