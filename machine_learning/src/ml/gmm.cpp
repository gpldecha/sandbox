#include "/home/guillaume/roscode/catkin_ws/src/machine_learning/include/machine_learning/ml/gmm.h"



Gmm::Gmm():varInName("evidence")
{

}


void Gmm::loadGmm()
{
	engEvalString(engine, "load bnet2");
	engEvalString(engine, "engine2 = jtree_inf_engine(bnet2);");
}


void Gmm::computeMarginal(const std::vector<double>& data){

			vector2matArr(data,Data_in);
			inputVar(varInName,Data_in);

			engEvalString(engine,"marginal = computeMarginal(engine2,evidence);");
			if ((mxData_out = engGetVariable(engine,"marginal")) != NULL){
					 Data_out = mxGetPr(mxData_out);
					 colMajorToRowMajor(Data_out,result,nbVar_out,nbSamples_out);
			}else{
					std::cout<< "unable to retrieve matlab output" << std::endl;
			}
}
