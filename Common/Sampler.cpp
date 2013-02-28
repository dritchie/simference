#include "Sampler.h"
#include <stan/mcmc/nuts.hpp>

using namespace std;
using namespace simference::Models;

namespace simference
{
	namespace Samplers
	{
		void Sampler::sample(std::vector<Sample>& samples,
							// How many iterations to run sampling for.
							int num_iterations,
							// How many of the above iterations count as 'warm-up' (samples discarded)
							int num_warmup,
							// Automatically choose step size during warm-up?
							bool epsilon_adapt,
							// Keep every how many samples?
							int num_thin,
							// Save the warm-up samples?
							bool save_warmup)
		{
			if (epsilon_adapt)
			{
				adaptOn();
			}
			for (int m = 0; m < num_iterations; ++m)
			{
				printf("Sampling iteration %d / %d\r", m+1, num_iterations);

				Sample sample = nextSample();

				if (m < num_warmup)
				{
					if (save_warmup && (m % num_thin) == 0)
					{
						samples.push_back(sample);
					} 
				}
				else 
				{
					if (epsilon_adapt && adapting())
					{
						adaptOff();
					}
					if (((m - num_warmup) % num_thin) != 0)
					{
						continue;
					}
					else 
					{
						samples.push_back(sample);
					}
				}
			}
			printf("\n");
		}


		// So we can restrict stan::mcmc::nuts to a single translation unit--
		// multiply-defined symbol errors will result otherwise.
		typedef boost::mt19937 DiffusionRNG;
		class DiffusionSamplerImpl : public stan::mcmc::nuts<DiffusionRNG> 
		{
		public:
			DiffusionSamplerImpl(Model& m, const vector<double>& initParams)
				: nuts(m, 10, -1, 0.0, true, 0.6, 0.05, DiffusionRNG((uint32_t)time(0)), &initParams)
			{}
			friend class DiffusionSampler;
		};

		DiffusionSampler::DiffusionSampler(StructurePtr s, Model& m, const vector<double>& initParams)
			: structure(s), implementation(new DiffusionSamplerImpl(m, initParams))
		{
		}

		DiffusionSampler::~DiffusionSampler()
		{
			delete implementation;
		}

		void DiffusionSampler::reinitialize(StructurePtr s, Model& m, const vector<double>& initParams)
		{
			structure = s;
			implementation->_model = m;
			implementation->set_params_r(initParams);
		}

		Sample DiffusionSampler::nextSample()
		{
			stan::mcmc::sample samp = implementation->next();
			return Sample(structure, samp.params_r(), samp.log_prob());
		}

		void DiffusionSampler::adaptOn()
		{
			implementation->adapt_on();
		}

		void DiffusionSampler::adaptOff()
		{
			implementation->adapt_off();
		}

		bool DiffusionSampler::adapting()
		{
			return implementation->adapting();
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

		void JumpSampler::adaptOn()
		{
			innerSampler->adaptOn();
		}

		void JumpSampler::adaptOff()
		{
			innerSampler->adaptOff();
		}

		bool JumpSampler::adapting()
		{
			return innerSampler->adapting();
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
