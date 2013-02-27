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
			currentUnrolledModel = templateModel->unroll(initStruct);
			innerSampler = DiffusionSamplerPtr(new DiffusionSampler(initStruct, *currentUnrolledModel, initParams));
		}

		Sample JumpSampler::nextSample()
		{
			// Choose whether to jump or diffuse, then execute the corresponding move
			if (Math::Probability::UniformDistribution<double>::Sample() < jumpFrequency)
			{
				return executeJumpMove();
			}
			else
			{
				Sample s = innerSampler->nextSample();
				currentParams = s.params;
				return s;
			}
		}

		Sample JumpSampler::executeJumpMove()
		{
			double currLp = currentUnrolledModel->log_prob(currentParams);

			// Propose new structure
			StructurePtr newStruct = jumpProposal();

			// Do dimension matching
			std::vector<double> matchedParams;
			DimensionMatchMap dimMatchMap;
			dimensionMatch(currentStruct, currentParams, newStruct, matchedParams, dimMatchMap);

			// Unroll factors for the current structure and new structure
			ModelPtr currModel, newModel, sharedModel;
			templateModel->unroll(currentStruct, newStruct, dimMatchMap, currModel, newModel, sharedModel);
			vector<ModelPtr> models;
			models.push_back(currModel);	// 0
			models.push_back(newModel);		// 1
			models.push_back(sharedModel);	// 2
			MixtureModel* mixModel = new MixtureModel(models);
			currentUnrolledModel = ModelPtr(mixModel);
			innerSampler->reinitialize(newStruct, *currentUnrolledModel, matchedParams);

			// Run the inner HMC kernel for numAnnealingSteps
			// Adjust the temperature of the factors each step
			// Accumulate log probability of each intermediate state
			vector<double>& weights = mixModel->getWeights();
			double annealingLpRatio = 0.0;
			double prevAnnealingLp = std::numeric_limits<double>::quiet_NaN();
			Sample lastAnnealingState;
			for (unsigned int i = 0; i < numAnnealingSteps; i++)
			{
				double temp = ((double)i)/(numAnnealingSteps-1);
				weights[0] = 1.0 - temp;
				weights[1] = temp;
				weights[2] = 1.0;
				lastAnnealingState = innerSampler->nextSample();
				double currAnnealingLp = lastAnnealingState.logprob;
				if (!isnan(prevAnnealingLp))
					annealingLpRatio += (prevAnnealingLp - currAnnealingLp);
				prevAnnealingLp = currAnnealingLp;
			}

			// Accept or reject the new structure
			const vector<double>& propParams = lastAnnealingState.params;
			double propLp = prevAnnealingLp;
			double forwardInitProposalLp, reverseInitProposalLp;
			if (dimMatchMap.direction == DimensionMatchMap::OldToNew)
			{
				forwardInitProposalLp = logProposalProbability(currentStruct, currentParams, newStruct, matchedParams);
				reverseInitProposalLp = logProposalProbability(newStruct, propParams, currentStruct, translateParameters(propParams, dimMatchMap));
			}
			else
			{
				forwardInitProposalLp = logProposalProbability(currentStruct, currentParams, newStruct, translateParameters(matchedParams, dimMatchMap));
				reverseInitProposalLp = logProposalProbability(newStruct, translateParameters(propParams, dimMatchMap), currentStruct, propParams);
			}
			double acceptLp = (propLp + reverseInitProposalLp) - (currLp - forwardInitProposalLp) + annealingLpRatio;
			if (log(Math::Probability::UniformDistribution<double>::Sample()) < acceptLp)
			{
				// Update state variables accordingly
				// (currentStruct, currentParams, currentUnrolledModel, innerSampler->_)
				currentStruct = newStruct;
				currentParams = propParams;
				currLp = propLp;
				currentUnrolledModel = templateModel->unroll(currentStruct);
				innerSampler->reinitialize(currentStruct, *currentUnrolledModel, currentParams);
			}
			return Sample(currentStruct, currentParams, currLp);
		}

		vector<double> JumpSampler::translateParameters(const vector<double>& params, const DimensionMatchMap& matching) const
		{
			vector<double> transp;
			for (unsigned int index : matching.paramIndexMap)
				transp.push_back(params[index]);
			return transp;
		}
	}
}