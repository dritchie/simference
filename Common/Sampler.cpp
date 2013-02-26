#include "Sampler.h"

using namespace std;
using namespace simference::Models;

namespace simference
{
	namespace Samplers
	{
		DiffusionSampler::DiffusionSampler(StructurePtr s, Model& m, const vector<double>& initParams)
			:
			nuts(m, 10, -1, 0.0, true, 0.6, 0.05, DiffusionRNG(time(0)), &initParams),
			structure(s)
		{}

		void DiffusionSampler::reinitialize(StructurePtr s, Model& m, const vector<double>& initParams)
		{
			structure = s;
			this->_model = m;
			this->set_params_r(initParams);
		}

		Sample DiffusionSampler::nextSample()
		{
			stan::mcmc::sample samp = next();	// Inherited from stan::mcmc::nuts
			return Sample(structure, samp.params_r(), samp.log_prob());
		}

		JumpSampler::JumpSampler(FactorTemplateModelPtr m, StructurePtr initStruct,
			const vector<double>& initParams,
			unsigned int nAnnealingSteps,
			double jumpFreq)
			:
		templateModel(m), currentStruct(initStruct), numAnnealingSteps(nAnnealingSteps),
			jumpFrequency(jumpFreq), currentParams(initParams)
		{
			vector<FactorPtr> initFactors;
			templateModel->unroll(initStruct, initFactors);
			currentUnrolledModel = ModelPtr(new FactorModel(initParams.size(), initFactors));
			innerSampler = DiffusionSamplerPtr(new DiffusionSampler(initStruct, *currentUnrolledModel, initParams));
		}

		Sample JumpSampler::nextSample()
		{
			// Choose whether to jump or diffuse, then execute the corresponding move
			if (Math::Probability::UniformDistribution<double>::Sample() < jumpFrequency)
			{
				return executeJumpMove();
			}
			else return innerSampler->nextSample();
		}

		Sample JumpSampler::executeJumpMove()
		{
			// Propose new structure
			StructurePtr newStruct = jumpProposal();

			// Do dimension matching
			unsigned int numCurrParams = currentStruct->numParams();
			unsigned int numNewParams = newStruct->numParams();
			if (numNewParams > numCurrParams)
			{
				vector<double> newParams;
				newStruct->getParams(newParams);
				unsigned int numDiff = numNewParams - numCurrParams;
				for (unsigned int i = numNewParams - numDiff; i < numNewParams; i++)
					currentParams.push_back(newParams[i]);
			}

			// Unroll factors for the current structure and new structure
			vector<FactorPtr> fCurr, fNew, fShared;
			templateModel->unroll(currentStruct, newStruct, fCurr, fNew, fShared);
			ModelPtr currModel = ModelPtr(new FactorModel(numNewParams, fCurr));
			ModelPtr newModel = ModelPtr(new FactorModel(numNewParams, fNew));
			ModelPtr sharedModel = ModelPtr(new FactorModel(numNewParams, fShared));
			vector<ModelPtr> models;
			models.push_back(currModel);	// 0
			models.push_back(newModel);		// 1
			models.push_back(sharedModel);	// 2
			MixtureModel* mixModel = new MixtureModel(models); // init with uniform weighting
			currentUnrolledModel = ModelPtr(mixModel);

			// Run the inner MCMC kernel for numAnnealingSteps
			// Adjust the temperature of the factors each step
			// Keep track of the log probability of each intermediate state
			vector<double>& weights = mixModel->getWeights();
			for (unsigned int i = 0; i < numAnnealingSteps; i++)
			{
				double temp = ((double)i)/(numAnnealingSteps-1);
				weights[0] = 1.0 - temp;
				weights[1] = temp;
				weights[2] = 1.0;
			}

			// Accept or reject the new structure
			// Update state variables accordingly
			// (currentStruct, currentParams, currentUnrolledModel, innerSampler->_)

			// return here
		}
	}
}