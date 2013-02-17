#include "MobileGrammar.h"
#include "../Common/Distributions.h"
#include <GL/glut.h>

using namespace std;

namespace simference
{
	namespace Grammar
	{
		namespace SimpleMobileGrammar
		{
			// Static initializer for production list
			vector<Production> StringEndpointVariable::InitProductionList()
			{
				vector<Production> p;
				// 1) Stick a terminal weight at the end of this string
				p.push_back(Production(
					// Conditional expression: this rule can always be applied
					[](const Variable& v) { return true; },
					// Probability expression: this rule gets more likely as depth increases
					[](const Variable& v)
					{
						StringEndpointVariable* sev = (StringEndpointVariable*)(&v);
						return sev->depth / (double)maxDepth;
					},
					// Unroll expression: replace the variable with a terminal weight with randomly-sampled mass
					[](const Variable& v)
					{
						vector<SymbolPtr> s;
						s.push_back(SymbolPtr(new WeightTerminal));
						return s;
					}
				));
				// 2) Stick a rod, plus the beginnings of its two branches, onto the end of this string
				p.push_back(Production(
					// Conditional expression: this rule can always be applied
					[](const Variable& v) { return true; },
					// Probability expression: this rule gets less likely as depth increases
					[](const Variable& v)
					{
						StringEndpointVariable* sev = (StringEndpointVariable*)(&v);
						return 1.0 - sev->depth / (double)maxDepth;
					},
					// Unroll expression
					[](const Variable& v)
					{
						StringEndpointVariable* sev = (StringEndpointVariable*)(&v);
						vector<SymbolPtr> s;
						// Rod
						s.push_back(SymbolPtr(new RodTerminal));
						// Left branch
						s.push_back(SymbolPtr(new StringTerminal(0)));
						s.push_back(SymbolPtr(new StringEndpointVariable(sev->depth+1)));
						// Right branch
						s.push_back(SymbolPtr(new StringTerminal(1)));
						s.push_back(SymbolPtr(new StringEndpointVariable(sev->depth+1)));
						return s;
					}
				));
				return p;
			};
			vector<Production> StringEndpointVariable::productionList = StringEndpointVariable::InitProductionList();



			DistribPtr StringTerminal::distribs[1] =
			{
				&stringLength
			};

			DistribPtr RodTerminal::distribs[2] =
			{
				&rodLength, &rodConnect
			};

			DistribPtr WeightTerminal::distribs[1] =
			{
				&weightRadius
			};
		}
	}
}