#include "/home/guillaume/roscode/catkin_ws/src/machine_learning/include/machine_learning/ml/ml.h"
#include "assert.h"

Ml::Ml(const std::vector<Finger>& fingers,vector<double>& probabilities):
fingers(fingers),
classifier(SVM),
counts_svm(5),
prob_buffer(5),
probabilities(probabilities)
{

	svm_interface.loadParameters("./models/model.txt");
	svm_interface.setDimensionOfInput(19);

	gmm.setDataSizeIn(1,19);
	gmm.setDataSizeOut(1,nbClass);
	gmm.startMatlab();
	gmm.loadGmm();
	runThread = false;
	first = true;
	E.resize(19);
	pressure = 0;
	y.resize(probabilities.size());

}

void Ml::setClassifier(ml_method classifier){
	this->classifier = classifier;
}

void Ml::setbrun(){
	runThread=true;
}

void Ml::run(){

	while(runThread){

		for(unsigned int e = 0; e < 19; e++){
			E[e] = fingers[FINGER_1].getEValue(e);
		}
		pressure = fingers[FINGER_1].getPdCValue();

		//std::cout<< "pressure: " << pressure << std::endl;
		if(pressure > 0.02){
			std::cout<< "not air" << std::endl;
			classify(E,classifier);
		}else{
			std::cout<< "air" << std::endl;
			classe = 4;
			counts_svm.push_back(classe);
			confidence_svm(counts_svm,confidence,nbClass);
			prob_buffer.push_back(tmp);
			writePorbClasses();
			exponential_mouving_average(probabilities,y,0.01);

			emit reUpdate();
		}
		msleep(10);
	}

}

void Ml::classify(const vector<double>& data, ml_method method){
	switch (method)
	{
	case SVM:
	{

		classe = svm_interface.svmPredict(data);
		counts_svm.push_back(classe);
		confidence_svm(counts_svm,confidence,nbClass);
		prob_buffer.push_back(tmp);

		break;
	}
	case HMM:
	{
		//matlab_interface_hmm.matlab_compare_hmm_Models(fingers[FINGER_1].data.standarised_E);

		break;
	}
	case GMM:
	{
		gmm.computeMarginal(data);
		tmp[0] = gmm.result[0][0];
		tmp[1] = gmm.result[0][1];
		tmp[2] = gmm.result[0][2];


		prob_buffer.push_back(tmp);
		break;
	}
	default:
	{
		std::cout<< "No such machine method available " << std::endl;
		break;
	}
	}

	writePorbClasses();
	exponential_mouving_average(probabilities,y,0.01);

	emit reUpdate();
}

void Ml::confidence_svm(const boost::circular_buffer<int>& counts, double *confidence, int nbClasses){

	for(int i = 0; i < nbClasses;i++){
		confidence[i] = 0;
	}
	for(unsigned int i = 0; i < counts.size(); i++){
		confidence[counts[i]-1] = confidence[counts[i]-1] + 1;
	}

	for(int i = 0; i < nbClasses;i++){
		confidence[i]=confidence[i]/counts.capacity();
		tmp[i] = confidence[i];
	}

}

int Ml::getClass(){
	return classe;
}

double Ml::getProbClass(int classe){
	if(prob_buffer.size() < 1){
		return 0;
	}else{
		return prob_buffer.back()[classe];
	}
}

inline void Ml::exponential_mouving_average(vector<double>& s,const vector<double>& y,double alpha ){
	if(first){
		s = y;
		first = false;
	}else{
		for(unsigned int i = 0; i < s.size();i++){
			s[i] = alpha * y[i] + (1 - alpha) * s[i];
		}
	}
}

inline void Ml::writePorbClasses(){
	for(unsigned int i = 0; i < y.size(); i++){
		y[i] = prob_buffer.back()[i];
	}
}

