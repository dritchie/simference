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