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
			Sample() : logprob(0.0) {}
			Sample(StructurePtr s, const std::vector<double>& p, double lp)
				: structure(s), params(p), logprob(lp) {}
			StructurePtr structure;
			std::vector<double> params;
			double logprob;
		};

		class Sampler
		{
		public:
			virtual Sample nextSample() = 0;
			virtual void adaptOn() = 0;
			virtual void adaptOff() = 0;
			virtual bool adapting() = 0;

			void sample(std::vector<Sample>& samples,
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
		private:
			DiffusionSamplerImpl* implementation;
			StructurePtr structure;
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

		protected:

			virtual StructurePtr jumpProposal() = 0;

			virtual void dimensionMatch(StructurePtr sFrom, const std::vector<double>& pFrom,
				StructurePtr sTo, std::vector<double>& pTo, DimensionMatchMap& matching) = 0;

			virtual double logProposalProbability(StructurePtr sFrom, const std::vector<double>& pFrom,
				StructurePtr sTo, const std::vector<double>& pTo) = 0;

			Sample executeJumpMove();

			std::vector<double> translateParameters(const std::vector<double>& params, const DimensionMatchMap& matching) const;

			DiffusionSamplerPtr innerSampler;
			Models::FactorTemplateModelPtr templateModel;
			Models::ModelPtr currentUnrolledModel;
			StructurePtr currentStruct;
			std::vector<double> currentParams;
			unsigned int numAnnealingSteps;
			double jumpFrequency;
		};
	}
}

#endif