#ifndef __GRAMMAR_INFERENCE_H
#define __GRAMMAR_INFERENCE_H

#include "Grammar.h"
#include "Sampler.h"
#include <unordered_set>

namespace simference
{
	namespace Models
	{
		class GrammarFactorTemplate : public FactorTemplate
		{
		public:
			void unroll(StructurePtr s, std::vector<FactorPtr>& factors) const;
			void unroll(StructurePtr sOld, StructurePtr sNew,
				std::vector<FactorPtr>& fOld, std::vector<FactorPtr>& fNew, std::vector<FactorPtr>& fShared) const;

			class Factor : public simference::Models::Factor
			{
			public:
				Factor(StructurePtr dtree,
					   const simference::Grammar::String<stan::agrad::var>::type & roots,
					   const std::unordered_set<simference::Grammar::SymbolPtr<stan::agrad::var>::type>& exclude);
				stan::agrad::var log_prob(const ParameterVector<stan::agrad::var>& params);
			private:
				std::vector<simference::Grammar::SymbolPtr<stan::agrad::var>::type> syms;
			};
		};
	}

	namespace Samplers
	{
		class GrammarJumpSampler : public JumpSampler
		{
		public:
			GrammarJumpSampler(Models::FactorTemplateModelPtr m, StructurePtr initStruct,
							   const std::vector<double>& initParams,
							   unsigned int nAnnealingSteps = 50,
							   double jumpFreq = 0.1)
			:
			JumpSampler(m, initStruct, initParams, nAnnealingSteps, jumpFreq),
			lastStructJumpedFrom(NULL), lastStructJumpedTo(NULL) {}

			// FOR TESTING
			StructurePtr jumpProposalTest() { return jumpProposal(); }
			void dimensionMatchTest(StructurePtr sFrom, const std::vector<double>& pFrom,
				StructurePtr sTo, std::vector<double>& pTo, DimensionMatchMap& matching)
			{ dimensionMatch(sFrom, pFrom, sTo, pTo, matching); }

		protected:
			StructurePtr jumpProposal();
			void dimensionMatch(StructurePtr sFrom, const std::vector<double>& pFrom,
				StructurePtr sTo, std::vector<double>& pTo, DimensionMatchMap& matching);
			double logProposalProbability(StructurePtr sFrom, const std::vector<double>& pFrom,
				StructurePtr sTo, const std::vector<double>& pTo);

		private:
			StructurePtr lastStructJumpedFrom;
			StructurePtr lastStructJumpedTo;
			double lastJumpForwardLp;
			double lastJumpReverseLp;
		};
	}
}

#endif