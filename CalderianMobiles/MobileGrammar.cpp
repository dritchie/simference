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
						double radius = SimpleMobileGrammar::weightRadius.sample();
						s.push_back(SymbolPtr(new WeightTerminal(radius)));
						return String(s, weightRadius.logprob(radius));
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
						double rodLen = rodLength.sample();
						double rodConn = rodConnect.sample();
						double leftStringLength = stringLength.sample();
						double rightStringLength = stringLength.sample();
						// Rod
						s.push_back(SymbolPtr(new RodTerminal(rodLen, rodLen*rodConn)));
						// Left branch
						s.push_back(SymbolPtr(new StringTerminal(leftStringLength)));
						s.push_back(SymbolPtr(new StringEndpointVariable(sev->depth+1)));
						// Right branch
						s.push_back(SymbolPtr(new StringTerminal(rightStringLength)));
						s.push_back(SymbolPtr(new StringEndpointVariable(sev->depth+1)));
						return String(s, rodLength.logprob(rodLen) + rodConnect.logprob(rodConn) +
							stringLength.logprob(leftStringLength) + stringLength.logprob(rightStringLength));
					}
				));
				return p;
			};
			vector<Production> StringEndpointVariable::productionList = StringEndpointVariable::InitProductionList();

			string StringEndpointVariable::print()
			{
				char buf[64];
				sprintf(buf, "SVar(%u)", depth);
				return string(buf);
			}

			string StringTerminal::print()
			{
				char buf[64];
				sprintf(buf, "String(%g)", length);
				return string(buf);
			}

			string RodTerminal::print()
			{
				char buf[64];
				sprintf(buf, "Rod(%g, %g)", length, stringConnectPoint);
				return string(buf);
			}

			string WeightTerminal::print()
			{
				char buf[64];
				sprintf(buf, "Weight(%g)", radius);
				return string(buf);
			}
		}
	}
}