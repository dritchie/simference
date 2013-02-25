#ifndef __SAMPLER_H
#define __SAMPLER_H

#include "Model.h"
#include "Distributions.h"
#include <stan/mcmc/nuts.hpp>

namespace simference
{
	namespace Samplers
	{
		template <class Structure>
		class Sample
		{
		public:
			typedef std::shared_ptr<Structure> StructurePtr;
			Sample() : logprob(0.0) {}
			Sample(StructurePtr s, const std::vector<double>& p, double lp)
				: structure(s), params(p), logprob(lp) {}
			StructurePtr structure;
			std::vector<double> params;
			double logprob;
		};

		template <class Structure>
		class Sampler
		{
		public:
			virtual Sample<Structure> nextSample() = 0;
		};

		typedef boost::mt19937 DiffusionRNG;
		template <class Structure>
		class DiffusionSampler : public Sampler<Structure>, public stan::mcmc::nuts<DiffusionRNG>
		{
		public:
			typedef std::shared_ptr<Structure> StructurePtr;
			DiffusionSampler(StructurePtr s, Models::Model& m, const std::vector<double>& initParams)
				:
				nuts(m, 10, -1, 0.0, true, 0.6, 0.05, DiffusionRNG(std::time(0)), &initParams),
				structure(s),
				model(m) {}
			Sample<Structure> nextSample()
			{
				stan::mcmc::sample samp = next();	// Inherited from stan::mcmc::nuts
				return Sample<Structure>(structure, samp.params_r(), samp.log_prob());
			}
		private:
			StructurePtr structure;
			Models::Model& model;
		};

		// Uses LARJ
		template <class Structure>
		class JumpSampler : public Sampler<Structure>
		{
		public:

			typedef std::shared_ptr<Structure> StructurePtr;

			JumpSampler(Models::FactorTemplateModel& m, StructurePtr initStruct,
				        const std::vector<double>& initParams, unsigned int nAnnealingSteps = 50)
				:
			model(m), currentStruct(initStruct), numAnnealingSteps(nAnnealingSteps),
			currentParams(initParams) {}

			void setCurrentStructure(StructurePtr s) { currentStruct = s; }

			// Propose a new structure, given the current one.
			virtual StructurePtr jumpProposal() = 0;

			Sample<Structure> nextSample()
			{
				// Propose new structure

				// Do dimension matching

				// Unroll factors for old structure and new structure

				// Run the inner MCMC kernel for numAnnealingSteps
				// Adjust the temperature of the factors each step
				// Keep track of the log probability of each intermediate state

				// Accept or reject the new structure
				// Set 'currentStruct' accordingly
			}

		protected:
			Models::FactorTemplateModel& model;
			StructurePtr currentStruct;
			std::vector<double> currentParams;
			unsigned int numAnnealingSteps;
		};

		template <class Structure>
		class MetaSampler : public Sampler<Structure>
		{
		public:

			typedef std::shared_ptr<Structure> StructurePtr;
			typedef std::shared_ptr<Sampler<Structure>> SamplerPtr;

			MetaSampler(const std::vector<SamplerPtr>& ss, const std::vector<double>& ws)
				: samplers(ss), weights(ws) {}

			Sample<Structure> next()
			{
				unsigned int which = Math::Probability::MultinomialDistribution<double>::Sample(samplerWeights);
				return samplers[which]->next();
			}

		private:
			std::vector<SamplerPtr> samplers;
			std::vector<double> samplerWeights;
		};
	}
}

#endif