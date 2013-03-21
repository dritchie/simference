#ifndef __SAMPLER_H
#define __SAMPLER_H

#include "Model.h"
#include "Distributions.h"

namespace simference
{
	namespace Samplers
	{
		class Sample
		{
		public:

			enum ProposalType
			{
				Diffusion = 0,
				JumpBegin,
				JumpEnd,
				Annealing
			};

			Sample() : logprob(0.0) {}
			Sample(StructurePtr s, const std::vector<double>& p, double lp, ProposalType pt, bool acc)
				: structure(s), params(p), logprob(lp), proposalType(pt), accepted(acc) {}
			void print(std::ostream& out) const;
			StructurePtr structure;
			std::vector<double> params;
			double logprob;
			ProposalType proposalType;
			bool accepted;
		};

		class Sampler
		{
		public:
			virtual Sample nextSample() = 0;
			virtual void adaptOn() = 0;
			virtual void adaptOff() = 0;
			virtual bool adapting() = 0;

			static void sample( Sampler& sampler,
								// Where to store generated samples
								std::vector<Sample>& samples,
								// How many iterations to run sampling for.
								int num_iterations = 1000,
								// How many of the above iterations count as 'warm-up' (samples discarded)
								int num_warmup = 100,
								// Automatically choose step size during warm-up?
								bool epsilon_adapt = true,
								// Keep every how many samples?
								int num_thin = 1,
								// Save the warm-up samples?
								bool save_warmup = false);
		};

		class DiffusionSamplerImpl;
		class DiffusionSampler : public Sampler
		{
		public:
			DiffusionSampler(StructurePtr s, Models::Model& m, const std::vector<double>& initParams);
			~DiffusionSampler();
			void reinitialize(StructurePtr s, Models::Model& m, const std::vector<double>& initParams);
			Sample nextSample();
			void adaptOn();
			void adaptOff();
			bool adapting();
			void writeAnalytics(std::ostream& out) const;
			static bool paramsEqual(const std::vector<double>& p1, const std::vector<double>& p2);
		private:
			DiffusionSamplerImpl* implementation;
			StructurePtr structure;
			
			// Analytics
			std::vector<double> prevParams;
			unsigned int numMovesAttempted;
			unsigned int numMovesAccepted;

			friend class JumpSampler;
		};

		typedef std::shared_ptr<DiffusionSampler> DiffusionSamplerPtr;

		// Uses LARJ
		class JumpSampler : public Sampler
		{
		public:

			JumpSampler(Models::FactorTemplateModelPtr m, StructurePtr initStruct,
				const std::vector<double>& initParams,
				unsigned int nAnnealingSteps = 50,
				double jumpFreq = 0.1);

			Sample nextSample();
			void adaptOn();
			void adaptOff();
			bool adapting();

			// This version doesn't have a warm-up period, and it uses
			// epsilon-adaptation throughout the whole run.
			static void sample(JumpSampler& sampler,
								std::vector<Sample>& samples,
								int num_iterations = 1000,
								int num_thin = 1);

			// This version uses a warm-up period with no jumps,
			// then turns out epsilon adaptation
			static void sampleWithWarmup(JumpSampler& sampler,
								std::vector<Sample>& samples,
								int num_iterations = 1000,
								int num_warmup = 100,
								bool epsilon_adapt = true,
								int num_thin = 1,
								bool save_warmup = false);

			// Analytics
			void writeAnalytics(std::ostream& out) const;
			double diffusionAcceptanceRatio() { return ((double)numDiffusionMovesAccepted)/numDiffusionMovesAttempted; }
			double annealingAcceptanceRatio() { return ((double)numAnnealingMovesAccepted)/numAnnealingMovesAttempted; }
			double jumpAcceptanceRatio() { return ((double)numJumpMovesAccepted)/numJumpMovesAttempted; }

		protected:

			virtual StructurePtr jumpProposal(std::vector<double>& extendedParams, DimensionMatchMap& dimMatchMap) = 0;

			virtual double logProposalProbability(StructurePtr sFrom, const std::vector<double>& pFrom,
				StructurePtr sTo, const std::vector<double>& pTo) = 0;

			Sample executeJumpMove();

			DiffusionSamplerPtr innerSampler;
			Models::FactorTemplateModelPtr templateModel;
			Models::ModelPtr currentUnrolledModel;
			StructurePtr currentStruct;
			std::vector<double> currentParams;
			unsigned int numAnnealingSteps;
			double jumpFrequency;

			// Analytics
			unsigned int numDiffusionMovesAttempted;
			unsigned int numDiffusionMovesAccepted;
			unsigned int numJumpMovesAttempted;
			unsigned int numJumpMovesAccepted;
			unsigned int numDiffDimJumpMovesAccepted;
			unsigned int numAnnealingMovesAttempted;
			unsigned int numAnnealingMovesAccepted;
			std::vector<Sample> annealingSamples;
		};
	}
}

#endif