#include "Sampler.h"
#include <stan/mcmc/nuts.hpp>

using namespace std;
using namespace simference::Models;

namespace simference
{
	namespace Samplers
	{
		void Sample::print(std::ostream& out) const
		{
			string propStr;
			if (proposalType == Jump)
				propStr = "Jump";
			if (proposalType == Diffusion)
				propStr = "Diffusion";
			if (proposalType == Annealing)
				propStr = "Annealing";
			out << "[Sample] Proposal: " << propStr << " | Accepted: " << accepted << " | LogProb: " << logprob << endl;
		}

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
			// This constructor assumes that epsilon adaptation is done
			DiffusionSamplerImpl(Model& m, const vector<double>& initParams, const DiffusionSamplerImpl& prev)
				: nuts(m, 10, /*prev._epsilon*/0.00074017, 0.0, false, 0.6, 0.05, prev._rand_int, &initParams)
			{}
			friend class DiffusionSampler;
		};

		DiffusionSampler::DiffusionSampler(StructurePtr s, Model& m, const vector<double>& initParams)
			: structure(s), implementation(new DiffusionSamplerImpl(m, initParams)), prevParams(initParams),
			numMovesAttempted(0), numMovesAccepted(0)
		{
		}

		DiffusionSampler::~DiffusionSampler()
		{
			delete implementation;
		}

		void DiffusionSampler::reinitialize(StructurePtr s, Model& m, const vector<double>& initParams)
		{
			// This feels so dirty, but it covers up a bug that I haven't been able to track down...
			stan::agrad::recover_memory();

			structure = s;
			auto oldimpl = implementation;
			implementation = new DiffusionSamplerImpl(m, initParams, *oldimpl);
			delete oldimpl;
			prevParams = initParams;
			numMovesAttempted = numMovesAccepted = 0;

			//// Spit out the current epsilon
			//cout << "epsilon: " << implementation->_epsilon << endl;
		}

		bool DiffusionSampler::paramsEqual(const std::vector<double>& p1, const std::vector<double>& p2)
		{
			for (unsigned int i = 0; i < p1.size(); i++)
			{
				if (p1[i] != p2[i])
					return false;
			}
			return true;
		}

		Sample DiffusionSampler::nextSample()
		{
			numMovesAttempted++;
			prevParams = implementation->_x;
			stan::mcmc::sample samp = implementation->next();
			bool moveAccepted = false;
			if (!paramsEqual(samp.params_r(), prevParams))
			{
				moveAccepted = true;
				numMovesAccepted++;
			}
			return Sample(structure, samp.params_r(), samp.log_prob(), Sample::Diffusion, moveAccepted);
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

		void DiffusionSampler::writeAnalytics(std::ostream& out) const
		{
			out << "-----------------------------------------------" << endl;
			out << "         DiffusionSampler Analytics            " << endl;
			out << "-----------------------------------------------" << endl;
			out << "	Attempted Moves: " << numMovesAttempted << endl;
			out << "	Accepted Moves:  " << numMovesAccepted << endl;
			out << "	Percentage:      " << ((double)numMovesAccepted)/numMovesAttempted << endl;
			out << "-----------------------------------------------" << endl;
			out << endl;
		}

		JumpSampler::JumpSampler(FactorTemplateModelPtr m, StructurePtr initStruct,
			const vector<double>& initParams,
			unsigned int nAnnealingSteps,
			double jumpFreq)
			:
		templateModel(m), currentStruct(initStruct), numAnnealingSteps(nAnnealingSteps),
			jumpFrequency(jumpFreq), currentParams(initParams),
			numDiffusionMovesAttempted(0), numDiffusionMovesAccepted(0),
			numJumpMovesAttempted(0), numJumpMovesAccepted(0), numDiffDimJumpMovesAccepted(0),
			numAnnealingMovesAttempted(0), numAnnealingMovesAccepted(0)
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
				numDiffusionMovesAttempted++;
				unsigned int prevNumAccepted = innerSampler->numMovesAccepted;
				Sample s = innerSampler->nextSample();
				numDiffusionMovesAccepted += (innerSampler->numMovesAccepted - prevNumAccepted);
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
			numJumpMovesAttempted++;

			double currLp = currentUnrolledModel->log_prob(currentParams);

			// Propose new structure and do dimension matching
			StructurePtr newStruct = jumpProposal();
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

			//// TEST: What if we just straight-up used the new model?
			////currentUnrolledModel = newModel;	// This leads to tons of weird behavior...
			//currentUnrolledModel = templateModel->unroll(newStruct);
			//vector<double> newp;
			//if (dimMatchMap.direction == DimensionMatchMap::OldToNew)
			//	newp = matchedParams;
			//else newp = translateParameters(matchedParams, dimMatchMap);
			//innerSampler->reinitialize(newStruct, *currentUnrolledModel, newp);
			//vector<int> dummy;
			//vector<double> gradient;
			//double test = currentUnrolledModel->grad_log_prob(matchedParams, dummy, gradient);

			// TEST: Compare the lp / gradient of 'currModel' with
			// the actual current model
			auto actualCurrModel = templateModel->unroll(currentStruct);
			vector<int> dummy;
			vector<double> grad1, grad2;
			double lp1 = actualCurrModel->grad_log_prob(currentParams, dummy, grad1);
			double lp2 = currModel->grad_log_prob(matchedParams, dummy, grad2);
			double lpdiff = lp1 - lp2;
			vector<double> gradiff(grad1.size());
			for (unsigned int i = 0; i < grad1.size(); i++) gradiff[i] = grad1[i] - grad2[i];

			// Run the inner HMC kernel for numAnnealingSteps
			// Adjust the temperature of the factors each step
			// Accumulate log probability of each intermediate state
			vector<double>& weights = mixModel->getWeights();
			double annealingLpRatio = 0.0;
			double prevAnnealingLp = std::numeric_limits<double>::quiet_NaN();
			Sample lastAnnealingState;
			annealingSamples.clear();
			for (unsigned int i = 0; i < numAnnealingSteps; i++)
			{
				double temp = ((double)i)/(numAnnealingSteps-1);
				weights[0] = 1.0 - temp;
				weights[1] = temp;
				weights[2] = 1.0;
				lastAnnealingState = innerSampler->nextSample();
				lastAnnealingState.proposalType = Sample::Annealing;
				annealingSamples.push_back(lastAnnealingState);
				double currAnnealingLp = lastAnnealingState.logprob;
				if (prevAnnealingLp == prevAnnealingLp)
					annealingLpRatio += (prevAnnealingLp - currAnnealingLp);
				prevAnnealingLp = currAnnealingLp;
			}
			numAnnealingMovesAttempted += innerSampler->numMovesAttempted;
			numAnnealingMovesAccepted += innerSampler->numMovesAccepted;

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
			double acceptLp = (propLp + reverseInitProposalLp) - (currLp + forwardInitProposalLp) + annealingLpRatio;
			bool jumpAccepted = false;
			if (log(Math::Probability::UniformDistribution<double>::Sample()) < acceptLp)
			{
				// Update state variables accordingly
				// (currentStruct, currentParams, currentUnrolledModel, innerSampler->_)

				if (!currentStruct->structurallyEquivalentTo(newStruct))
					numDiffDimJumpMovesAccepted++;

				currentStruct = newStruct;
				if (dimMatchMap.direction == DimensionMatchMap::OldToNew)
					currentParams = propParams;
				else
					currentParams = translateParameters(propParams, dimMatchMap);
				currLp = propLp;
				numJumpMovesAccepted++;
				jumpAccepted = true;
			}

			currentUnrolledModel = templateModel->unroll(currentStruct);
			innerSampler->reinitialize(currentStruct, *currentUnrolledModel, currentParams);

			return Sample(currentStruct, currentParams, currLp, Sample::Jump, jumpAccepted);
		}

		vector<double> JumpSampler::translateParameters(const vector<double>& params, const DimensionMatchMap& matching) const
		{
			vector<double> transp;
			for (unsigned int index : matching.paramIndexMap)
				transp.push_back(params[index]);
			return transp;
		}

		void JumpSampler::sample(vector<Sample>& samples,
								int num_iterations,
								int num_warmup ,
								bool epsilon_adapt,
								int num_thin,
								bool save_warmup)
		{
			if (epsilon_adapt)
			{
				adaptOn();
			}
			for (int m = 0; m < num_iterations; ++m)
			{
				printf("Sampling iteration %d / %d\r", m+1, num_iterations);

				if (m < num_warmup)
				{
					// Don't start dimension jumping until after warm-up.
					Sample sample = innerSampler->nextSample();
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

					unsigned int numJumps = numJumpMovesAttempted;
					Sample sample = nextSample();

					if (((m - num_warmup) % num_thin) != 0)
					{
						continue;
					}
					else 
					{
						// If we just jumped, then we should splice in the annealing samples, too.
						if (numJumpMovesAttempted > numJumps)
						{
							samples.insert(samples.end(), annealingSamples.begin(), annealingSamples.end());
						}
						samples.push_back(sample);
					}
				}
			}
			printf("\n");
		}

		void JumpSampler::writeAnalytics(std::ostream& out) const
		{
			out << "-----------------------------------------------" << endl;
			out << "           JumpSampler Analytics               " << endl;
			out << "-----------------------------------------------" << endl;
			out << " Diffusion Stats:" << endl;
			out << "	Attempted Moves: " << numDiffusionMovesAttempted << endl;
			out << "	Accepted Moves:  " << numDiffusionMovesAccepted << endl;
			out << "	Percentage:      " << ((double)numDiffusionMovesAccepted)/numDiffusionMovesAttempted << endl;
			out << "-----------------------------------------------" << endl;
			out << " Annealing Stats:" << endl;
			out << "	Attempted Moves: " << numAnnealingMovesAttempted << endl;
			out << "	Accepted Moves:  " << numAnnealingMovesAccepted << endl;
			out << "	Percentage:      " << ((double)numAnnealingMovesAccepted)/numAnnealingMovesAttempted << endl;
			out << "-----------------------------------------------" << endl;
			out << " Jump Stats:" << endl;
			out << "	Attempted Moves: " << numJumpMovesAttempted << endl;
			out << "	Accepted Moves:  " << numJumpMovesAccepted << endl;
			out << "	Percentage:      " << ((double)numJumpMovesAccepted)/numJumpMovesAttempted << endl;
			out << "	Accepted Diff Struct Moves:  " << numDiffDimJumpMovesAccepted << endl;
			out << "	Percentage:      " << ((double)numDiffDimJumpMovesAccepted)/numJumpMovesAttempted << endl;
			out << "-----------------------------------------------" << endl;
			out << endl;
		}
	}
}
