#ifndef __MOBILE_GRAMMAR_H
#define __MOBILE_GRAMMAR_H

#include "../Common/PCFG.h"

namespace simference
{
	namespace Grammar
	{
		namespace MobileGrammar
		{
			template <typename RealNum>
			class Parameters
			{
			public:
				static TruncatedNormalDistribution<RealNum> stringLength;
				static TruncatedNormalDistribution<RealNum> rodLength;
				static TruncatedNormalDistribution<RealNum> rodConnect;
				static TruncatedNormalDistribution<RealNum> weightRadius;
				static unsigned int maxDepth;
			};
			template<typename RealNum>
			TruncatedNormalDistribution<RealNum> Parameters<RealNum>::stringLength(2.0, 0.5, 0.0, 10.0);
			template<typename RealNum>
			TruncatedNormalDistribution<RealNum> Parameters<RealNum>::rodLength(3.0, 1.0, 0.0, 10.0);
			template<typename RealNum>
			TruncatedNormalDistribution<RealNum> Parameters<RealNum>::rodConnect(1.5, 0.45, 0.0, 3.0);
			template<typename RealNum>
			TruncatedNormalDistribution<RealNum> Parameters<RealNum>::weightRadius(0.5, 0.25, 0.0, 10.0);
			template<typename RealNum>
			unsigned int Parameters<RealNum>::maxDepth = 5;

			template <typename RealNum>
			class StringTerminal : public GeneralTerminal<RealNum, 1>
			{
			public:
				StringTerminal(unsigned int id) : GeneralTerminal(distribs), index(id) {}
				char* name() { return "String"; }
				unsigned int index;
				static Distribution<RealNum>* distribs[1];
			};
			template<typename RealNum>
			Distribution<RealNum>* StringTerminal<RealNum>::distribs[1] = { &Parameters<RealNum>::stringLength };

			enum StringTerminalParamIndices { StringLength = 0 };

			template <typename RealNum>
			class RodTerminal : public GeneralTerminal<RealNum, 2>
			{
			public:
				RodTerminal() : GeneralTerminal(distribs) {}
				char* name() { return "Rod"; }
				static Distribution<RealNum>* distribs[2];
			};
			template<typename RealNum>
			Distribution<RealNum>* RodTerminal<RealNum>::distribs[2] = { &Parameters<RealNum>::rodLength, &Parameters<RealNum>::rodConnect };
			enum RodTerminalParamIndices { RodLength = 0, RodConnectPoint };

			template <typename RealNum>
			class WeightTerminal : public GeneralTerminal<RealNum, 1>
			{
			public:
				WeightTerminal() : GeneralTerminal(distribs) {}
				char* name() { return "Weight"; }
				static Distribution<RealNum>* distribs[1];
			};
			template<typename RealNum>
			Distribution<RealNum>* WeightTerminal<RealNum>::distribs[1] = { &Parameters<RealNum>::weightRadius };
			enum WeightTerminalParamIndices { WeightRadius = 0 };

			template <typename RealNum>
			class StringEndpointVariable : public Variable<RealNum>
			{
			public:

				StringEndpointVariable(unsigned int d) : depth(d) {}

				static std::vector<Production<RealNum>> InitProductionList()
				{
					std::vector<Production<RealNum>> p;
					// 1) Stick a terminal weight at the end of this string
					p.push_back(Production<RealNum>(
						// Conditional expression: this rule can always be applied
						[](const Variable<RealNum>& v) { return true; },
						// Probability expression: this rule gets more likely as depth increases
						[](const Variable<RealNum>& v)
					{
						StringEndpointVariable<RealNum>* sev = (StringEndpointVariable<RealNum>*)(&v);
						return sev->depth / (double)(Parameters<RealNum>::maxDepth);
					},
						// Unroll expression: replace the variable with a terminal weight with randomly-sampled mass
						[](const Variable<RealNum>& v)
					{
						std::vector<SymbolPtr> s;
						s.push_back(SymbolPtr(new WeightTerminal<RealNum>));
						return s;
					}
					));
					// 2) Stick a rod, plus the beginnings of its two branches, onto the end of this string
					p.push_back(Production<RealNum>(
						// Conditional expression: this rule can always be applied
						[](const Variable<RealNum>& v) { return true; },
						// Probability expression: this rule gets less likely as depth increases
						[](const Variable<RealNum>& v)
					{
						StringEndpointVariable<RealNum>* sev = (StringEndpointVariable<RealNum>*)(&v);
						return 1.0 - sev->depth / (double)(Parameters<RealNum>::maxDepth);
					},
						// Unroll expression
						[](const Variable<RealNum>& v)
					{
						StringEndpointVariable<RealNum>* sev = (StringEndpointVariable<RealNum>*)(&v);
						std::vector<SymbolPtr> s;
						// Rod
						s.push_back(SymbolPtr(new RodTerminal<RealNum>));
						// Left branch
						s.push_back(SymbolPtr(new StringTerminal<RealNum>(0)));
						s.push_back(SymbolPtr(new StringEndpointVariable<RealNum>(sev->depth+1)));
						// Right branch
						s.push_back(SymbolPtr(new StringTerminal<RealNum>(1)));
						s.push_back(SymbolPtr(new StringEndpointVariable<RealNum>(sev->depth+1)));
						return s;
					}
					));
					return p;
				}

				const std::vector<Production<RealNum>>& productions() { return productionList; }

				void print(std::ostream& outstream) { outstream << "SVar(" << depth << ")"; }

				unsigned int depth;
				static std::vector<Production<RealNum>> productionList;
			};

			template<typename RealNum>
			std::vector<Production<RealNum>> StringEndpointVariable<RealNum>::productionList = StringEndpointVariable<RealNum>::InitProductionList();
		}
	}
}

#endif