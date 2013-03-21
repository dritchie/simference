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
			if (proposalType == JumpBegin)
				propStr = "JumpBegin";
			if (proposalType == JumpEnd)
				propStr = "JumpEnd";
			if (proposalType == Diffusion)
				propStr = "Diffusion";
			if (proposalType == Annealing)
				propStr = "Annealing";
			out << "[Sample] Proposal: " << propStr << " | Accepted: " << accepted << " | LogProb: " << logprob << endl;
		}

		void Sampler::sample(Sampler& sampler,
							// Where to store samples
							std::vector<Sample>& samples,
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
				sampler.adaptOn();
			}
			for (int m = 0; m < num_iterations; ++m)
			{
				printf("Sampling iteration %d / %d\r", m+1, num_iterations);

				Sample sample = sampler.nextSample();

				if (m < num_warmup)
				{
					if (save_warmup && (m % num_thin) == 0)
					{
						samples.push_back(sample);
					} 
				}
				else 
				{
					if (epsilon_adapt && sampler.adapting())
					{
						sampler.adaptOff();
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
			DiffusionSamplerImpl(Model& m, const vector<double>& initParams, const DiffusionSamplerImpl& prev)
				: nuts(m, 10, prev._epsilon, prev._epsilon_pm, prev._epsilon_adapt, prev._delta, prev._gamma, prev._rand_int, &initParams)
			{
			}
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

			stan::agrad::recover_memory();
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
			DimensionMatchMap dimMatchMap;
			std::vector<double> extendedParams;
			StructurePtr newStruct = jumpProposal(extendedParams, dimMatchMap);

			// Unroll factors for the current structure and new structure
			ModelPtr currModel, newModel, sharedModel;
			templateModel->unroll(currentStruct, newStruct, dimMatchMap, currModel, newModel, sharedModel);
			vector<ModelPtr> models;
			models.push_back(currModel);	// 0
			models.push_back(newModel);		// 1
			models.push_back(sharedModel);	// 2
			MixtureModel* mixModel = new MixtureModel(models);
			vector<double>& weights = mixModel->getWeights();
			weights[0] = 1.0;
			weights[1] = 0.0;
			weights[2] = 1.0;
			currentUnrolledModel = ModelPtr(mixModel);
			innerSampler->reinitialize(newStruct, *currentUnrolledModel, extendedParams);

			// Run the inner HMC kernel for numAnnealingSteps
			// Adjust the temperature of the factors each step
			// Accumulate probability ratio as we go
			annealingSamples.clear();
			annealingSamples.push_back(Sample(newStruct, extendedParams, currLp, Sample::JumpBegin, true));
			double annealingLpRatio = 0.0;
			for (unsigned int i = 0; i < numAnnealingSteps; i++)
			{
				double alpha = ((double)i)/numAnnealingSteps;
				weights[0] = 1.0 - alpha;
				weights[1] = alpha;
				weights[2] = 1.0;

				unsigned int prevNumAccept = innerSampler->numMovesAccepted;
				auto samp = innerSampler->nextSample();
				unsigned int currNumAccept = innerSampler->numMovesAccepted;
				samp.proposalType = Sample::Annealing;
				if (currNumAccept == prevNumAccept)
				{
					// If we rejected this proposal, we need to re-compute its log probability, since
					// the fixed-dimension inner sampler will give us the last accepted probability, which is
					// outdated due to annealing interpolation.
					samp.logprob = currentUnrolledModel->log_prob(samp.params);
				}

				vector<double>& xt = annealingSamples.back().params;
				double lpt = currentUnrolledModel->log_prob(xt);
				double lptplus1 = samp.logprob;
				annealingLpRatio += (lpt - lptplus1);

				annealingSamples.push_back(samp);
			}
			numAnnealingMovesAttempted += innerSampler->numMovesAttempted;
			numAnnealingMovesAccepted += innerSampler->numMovesAccepted;

			weights[0] = 0.0;
			weights[1] = 1.0;
			weights[2] = 1.0;
			vector<double>& propParams = annealingSamples.back().params;
			double propLp = currentUnrolledModel->log_prob(propParams);

			// Accept or reject the new structure
			double forwardInitProposalLp, reverseInitProposalLp;
			forwardInitProposalLp = logProposalProbability(currentStruct, currentParams, newStruct, dimMatchMap.translateExtendedToNew(extendedParams));
			reverseInitProposalLp = logProposalProbability(newStruct, dimMatchMap.translateExtendedToNew(propParams), currentStruct, dimMatchMap.translateExtendedToOld(propParams));
			double acceptLp = (propLp + reverseInitProposalLp) - (currLp + forwardInitProposalLp) + annealingLpRatio;
			bool jumpAccepted = false;
			if (log(Math::Probability::UniformDistribution<double>::Sample()) < acceptLp)
			{
				// Update state variables accordingly
				if (!currentStruct->structurallyEquivalentTo(newStruct))
					numDiffDimJumpMovesAccepted++;
				currentStruct = newStruct;
				currentParams = dimMatchMap.translateExtendedToNew(propParams);
				currLp = propLp;
				numJumpMovesAccepted++;
				jumpAccepted = true;
			}

			//// TEST: While we're debugging, just force acceptance for all jumps
			//bool jumpAccepted = true;
			//currentStruct = newStruct;
			//currentParams = dimMatchMap.translateExtendedToNew(propParams);
			//currLp = propLp;
			//numJumpMovesAccepted++;
			//if (!currentStruct->structurallyEquivalentTo(newStruct))
			//	numDiffDimJumpMovesAccepted++;

			currentUnrolledModel = templateModel->unroll(currentStruct);
			innerSampler->reinitialize(currentStruct, *currentUnrolledModel, currentParams);

			// Translate the parameters of all the annealing samples so we can analyze/visualize them later
			for (Sample& s : annealingSamples)
				s.params = dimMatchMap.translateExtendedToNew(s.params);

			return Sample(currentStruct, currentParams, currLp, Sample::JumpEnd, jumpAccepted);
		}

		void JumpSampler::sample(JumpSampler& sampler,
								vector<Sample>& samples,
								int num_iterations,
								int num_thin)
		{
			// We must use adaptation all the time, because
			// the optimal epsilon changes as we jump into different subspaces.
			sampler.adaptOn();

			for (int m = 0; m < num_iterations; ++m)
			{
				printf("Sampling iteration %d / %d\r", m+1, num_iterations);

				unsigned int numJumps = sampler.numJumpMovesAttempted;
				Sample sample = sampler.nextSample();

				if ((m % num_thin) != 0)
				{
					continue;
				}
				else 
				{
					// If we just jumped, then we should splice in the annealing samples, too.
					if (sampler.numJumpMovesAttempted > numJumps)
					{
						samples.insert(samples.end(), sampler.annealingSamples.begin(), sampler.annealingSamples.end());
					}
					samples.push_back(sample);
				}
			}
			printf("\n");
		}

		void JumpSampler::sampleWithWarmup(JumpSampler& sampler,
								vector<Sample>& samples,
								int num_iterations,
								int num_warmup ,
								bool epsilon_adapt,
								int num_thin,
								bool save_warmup)
		{
			// Remember the correct jump probability
			double jumpProb = sampler.jumpFrequency;

			if (epsilon_adapt)
			{
				sampler.adaptOn();
				sampler.jumpFrequency = 0.0;
			}
			for (int m = 0; m < num_iterations; ++m)
			{
				printf("Sampling iteration %d / %d\r", m+1, num_iterations);

				if (m < num_warmup)
				{
					Sample sample = sampler.nextSample();
					if (save_warmup && (m % num_thin) == 0)
					{
						samples.push_back(sample);
					} 
				}
				else 
				{
					if (epsilon_adapt && sampler.adapting())
					{
						sampler.adaptOff();
						sampler.jumpFrequency = jumpProb;
					}

					unsigned int numJumps = sampler.numJumpMovesAttempted;
					Sample sample = sampler.nextSample();

					if (((m - num_warmup) % num_thin) != 0)
					{
						continue;
					}
					else 
					{
						// If we just jumped, then we should splice in the annealing samples, too.
						if (sampler.numJumpMovesAttempted > numJumps)
						{
							samples.insert(samples.end(), sampler.annealingSamples.begin(), sampler.annealingSamples.end());
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
