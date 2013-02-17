#ifndef __MOBILE_GRAMMAR_H
#define __MOBILE_GRAMMAR_H

#include "../Common/PCFG.h"
#include "../Common/Distributions.h"

namespace simference
{
	namespace Grammar
	{
		/** 
		This captures only rigid structures that use bipodial branching
		with strings and rods.
		**/
		namespace SimpleMobileGrammar
		{
			// Parameters
			static simference::Math::Probability::TruncatedNormalDistribution<double> stringLength(2.0, 0.5, 0.0, 10.0);
			static simference::Math::Probability::TruncatedNormalDistribution<double> rodLength(3.0, 1.0, 0.0, 10.0);
			static simference::Math::Probability::TruncatedNormalDistribution<double> rodConnect(1.5, 0.45, 0.0, 3.0);
			static simference::Math::Probability::TruncatedNormalDistribution<double> weightRadius(0.5, 0.25, 0.0, 10.0);
			static unsigned int maxDepth = 5;

			class StringEndpointVariable : public Variable
			{
			public:
				StringEndpointVariable(unsigned int d) : depth(d) {}
				unsigned int depth;
				static std::vector<Production> productionList;
				static std::vector<Production> InitProductionList();
				const std::vector<Production>& productions() { return productionList; }
				void print(std::ostream& outstream) { outstream << "SVar(" << depth << ")"; }
			};

			class StringTerminal : public GeneralTerminal<1>
			{
			public:
				StringTerminal(unsigned int id) : GeneralTerminal(distribs), index(id) {}
				char* name() { return "String"; }
				static DistribPtr distribs[1];
				unsigned int index;
			};

			class RodTerminal : public GeneralTerminal<2>
			{
			public:
				RodTerminal() : GeneralTerminal(distribs) {}
				char* name() { return "Rod"; }
				static DistribPtr distribs[2];
			};

			class WeightTerminal : public GeneralTerminal<1>
			{
			public:
				WeightTerminal() : GeneralTerminal(distribs) {}
				char* name() { return "Weight"; }
				static DistribPtr distribs[1];
			};
		}
	}
}

#endif