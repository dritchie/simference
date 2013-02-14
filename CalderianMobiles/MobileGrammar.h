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
			static simference::Math::Probability::TruncatedNormalDistribution<double> rodConnect(0.5, 0.15, 0.0, 1.0);
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
				std::string print();
			};

			class StringTerminal : public Terminal
			{
			public:
				StringTerminal(double l, unsigned int id) : length(l), index(id) {}
				std::string print();
				double length;
				unsigned int index;
			};

			class RodTerminal : public Terminal
			{
			public:
				RodTerminal(double l, double scp) : length(l), stringConnectPoint(scp) {}
				std::string print();
				double length;
				double stringConnectPoint;
			};

			class WeightTerminal : public Terminal
			{
			public:
				WeightTerminal(double r) : radius(r) {}
				std::string print();
				double radius;
			};
		}
	}
}

#endif