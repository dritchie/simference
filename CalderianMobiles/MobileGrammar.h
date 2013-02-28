#ifndef __MOBILE_GRAMMAR_H
#define __MOBILE_GRAMMAR_H

#include "../Common/Grammar.h"

namespace simference
{
	namespace Grammar
	{
		namespace MobileGrammar
		{
			template <typename RealNum>
			class Parameters
			{
			private:
				Parameters() :
					stringLength(2.0, 0.5, 0.0, 10.0),
					rodLength(3.0, 1.0, 0.0, 10.0),
					rodConnect(0.5, 0.15, 0.0, 1.0),
					weightRadius(0.5, 0.25, 0.0, 10.0),
					//stringLength(2.0, 0.5),
					//rodLength(3.0, 1.0),
					//rodConnect(0.5, 0.15),
					//weightRadius(0.5, 0.25),
					maxDepth(5) {}
				static Parameters<RealNum>* instance;

			public:
				TruncatedNormalDistribution<RealNum, double> stringLength;
				TruncatedNormalDistribution<RealNum, double> rodLength;
				TruncatedNormalDistribution<RealNum, double> rodConnect;
				TruncatedNormalDistribution<RealNum, double> weightRadius;
				//NormalDistribution<RealNum, double> stringLength;
				//NormalDistribution<RealNum, double> rodLength;
				//NormalDistribution<RealNum, double> rodConnect;
				//NormalDistribution<RealNum, double> weightRadius;
				unsigned int maxDepth;
				static Parameters* Instance() { if (instance == NULL) {instance = new Parameters<RealNum>;} return instance;}
			};
			template <typename RealNum> Parameters<RealNum>* Parameters<RealNum>::instance = NULL;

			template <typename RealNum>
			class StringTerminal : public GeneralTerminal<RealNum, 1>
			{
			public:
				StringTerminal(unsigned int depth, unsigned int id) : GeneralTerminal(depth, GetDistribs()), index(id) {}
				char* name() const { return "String"; }
				unsigned int index;
				static Distribution<RealNum>* distribs[1];
				static Distribution<RealNum>** GetDistribs()
				{
					if (distribs[0] == NULL) { distribs[0] = &Parameters<RealNum>::Instance()->stringLength; }
					return distribs;
				}
			};
			template <typename RealNum> Distribution<RealNum>* StringTerminal<RealNum>::distribs[1] = {NULL};
			enum StringTerminalParamIndices { StringLength = 0 };

			template <typename RealNum>
			class RodTerminal : public GeneralTerminal<RealNum, 2>
			{
			public:
				RodTerminal(unsigned int depth) : GeneralTerminal(depth, GetDistribs()) {}
				char* name() const { return "Rod"; }
				static Distribution<RealNum>* distribs[2];
				static Distribution<RealNum>** GetDistribs()
				{
					if (distribs[0] == NULL) { distribs[0] = &Parameters<RealNum>::Instance()->rodLength; distribs[1] = &Parameters<RealNum>::Instance()->rodConnect; }
					return distribs;
				}
			};
			template<typename RealNum> Distribution<RealNum>* RodTerminal<RealNum>::distribs[2] = { NULL, NULL};
			enum RodTerminalParamIndices { RodLength = 0, RodConnectPoint };

			template <typename RealNum>
			class WeightTerminal : public GeneralTerminal<RealNum, 1>
			{
			public:
				WeightTerminal(unsigned int depth) : GeneralTerminal(depth, GetDistribs()) {}
				char* name() const { return "Weight"; }
				static Distribution<RealNum>* distribs[1];
				static Distribution<RealNum>** GetDistribs()
				{
					if (distribs[0] == NULL) { distribs[0] = &Parameters<RealNum>::Instance()->weightRadius; }
					return distribs;
				}
			};
			template<typename RealNum> Distribution<RealNum>* WeightTerminal<RealNum>::distribs[1] = { NULL };
			enum WeightTerminalParamIndices { WeightRadius = 0 };

			template <typename RealNum>
			class StringEndpointVariable : public Variable<RealNum>
			{
			public:

				StringEndpointVariable(unsigned int depth) : Variable(depth) {}

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
						return v.depth / (RealNum)(Parameters<RealNum>::Instance()->maxDepth);
					},
						// Unroll expression: replace the variable with a terminal weight with randomly-sampled mass
						[](const Variable<RealNum>& v)
					{
						std::vector<SymbolPtr> s;
						s.push_back(SymbolPtr(new WeightTerminal<RealNum>(v.depth+1)));
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
						return 1.0 - v.depth / (RealNum)(Parameters<RealNum>::Instance()->maxDepth);
					},
						// Unroll expression
						[](const Variable<RealNum>& v)
					{
						std::vector<SymbolPtr> s;
						// Rod
						s.push_back(SymbolPtr(new RodTerminal<RealNum>(v.depth+1)));
						// Left branch
						s.push_back(SymbolPtr(new StringTerminal<RealNum>(v.depth+1, 0)));
						s.push_back(SymbolPtr(new StringEndpointVariable<RealNum>(v.depth+1)));
						// Right branch
						s.push_back(SymbolPtr(new StringTerminal<RealNum>(v.depth+1, 1)));
						s.push_back(SymbolPtr(new StringEndpointVariable<RealNum>(v.depth+1)));
						return s;
					}
					));
					return p;
				}

				const std::vector<Production<RealNum>>& productions() { return productionList; }

				void print(std::ostream& outstream) const { outstream << "SVar(" << depth << ")"; }

				static std::vector<Production<RealNum>> productionList;
			};

			template<typename RealNum>
			std::vector<Production<RealNum>> StringEndpointVariable<RealNum>::productionList = StringEndpointVariable<RealNum>::InitProductionList();
		}
	}
}

#endif