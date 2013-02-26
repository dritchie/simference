#ifndef __SAMPLER_H
#define __SAMPLER_H

#include "Model.h"
#include "Distributions.h"
#include <stan/mcmc/nuts.hpp>

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
		};

		typedef boost::mt19937 DiffusionRNG;
		class DiffusionSampler : public Sampler, public stan::mcmc::nuts<DiffusionRNG>
		{
		public:
			DiffusionSampler(StructurePtr s, Models::Model& m, const std::vector<double>& initParams);
			void reinitialize(StructurePtr s, Models::Model& m, const std::vector<double>& initParams);
			Sample nextSample();
		private:
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

		protected:

			// Propose a new structure, given the current one.
			virtual StructurePtr jumpProposal() = 0;
			//TODO: 'logProposalProbability' method (takes two structures, two param lists)

			Sample executeJumpMove();

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