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
						double mass = SimpleMobileGrammar::stringLength.sample();
						s.push_back(SymbolPtr(new WeightTerminal(mass)));
						return String(s, stringLength.logprob(mass));
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
						s.push_back(SymbolPtr(new BranchBeginTerminal));
							s.push_back(SymbolPtr(new StringTerminal(leftStringLength)));
							s.push_back(SymbolPtr(new StringEndpointVariable(sev->depth+1)));
						s.push_back(SymbolPtr(new BranchEndTerminal));
						// Right branch
						s.push_back(SymbolPtr(new BranchBeginTerminal));
							s.push_back(SymbolPtr(new StringTerminal(rightStringLength)));
							s.push_back(SymbolPtr(new StringEndpointVariable(sev->depth+1)));
						s.push_back(SymbolPtr(new BranchEndTerminal));
						return String(s, rodLength.logprob(rodLen) + rodConnect.logprob(rodConn) +
							stringLength.logprob(leftStringLength) + stringLength.logprob(rightStringLength));
					}
				));
				return p;
			};
			vector<Production> StringEndpointVariable::productionList = StringEndpointVariable::InitProductionList();

			#define SET_STRING_COLOR (glColor3f(0.5f, 0.5f, 0.5f))
			#define SET_ROD_COLOR (glColor3f(0.2f, 0.2f, 0.2f))
			#define SET_WEIGHT_COLOR (glColor3f(0.2f, 0.2f, 1.0f))
			#define STRING_WIDTH 1.0f
			#define ROD_WIDTH 3.0f
			#define WEIGHT_SLICES 16

			string StringEndpointVariable::print()
			{
				char buf[64];
				sprintf(buf, "SVar(%u)", depth);
				return string(buf);
			}

			void StringTerminal::render()
			{
				// Draw a line
				glLineWidth(STRING_WIDTH);
				SET_STRING_COLOR;
				glMatrixMode(GL_MODELVIEW);
				glBegin(GL_LINES);
					glVertex2f(0.0f, 0.0f);
					glVertex2f(0.0f, -this->length);
				glEnd();

				// Move down to the bottom of the string
				glTranslatef(0.0f, -this->length, 0.0f);
			}

			string StringTerminal::print()
			{
				char buf[64];
				sprintf(buf, "String(%g)", length);
				return string(buf);
			}

			void RodTerminal::render()
			{
				// Draw a rod
				glLineWidth(ROD_WIDTH);
				SET_ROD_COLOR;
				glMatrixMode(GL_MODELVIEW);
				glBegin(GL_LINES);
					glVertex2f(-stringConnectPoint, 0.0f);
					glVertex2f(length-stringConnectPoint, 0.0f);
				glEnd();

				// Push transformation for right branch
				glPushMatrix();
				glTranslatef(-stringConnectPoint, 0.0f, 0.0f);

				// Push transformation for left branch
				glPushMatrix();
				glTranslatef(length, 0.0f, 0.0f);
			}

			string RodTerminal::print()
			{
				char buf[64];
				sprintf(buf, "Rod(%g, %g)", length, stringConnectPoint);
				return string(buf);
			}

			void WeightTerminal::render()
			{
				// Push down by the size (which is interpreted as radius)
				glTranslatef(0.0f, -mass, 0.0f);

				// Draw a weight as a sphere
				glutSolidSphere(mass, WEIGHT_SLICES, WEIGHT_SLICES);
			}

			string WeightTerminal::print()
			{
				char buf[64];
				sprintf(buf, "Weight(%g)", mass);
				return string(buf);
			}

			void BranchBeginTerminal::render()
			{
			}

			string BranchBeginTerminal::print()
			{
				return "[";
			}

			void BranchEndTerminal::render()
			{
				glMatrixMode(GL_MODELVIEW);
				glPopMatrix();
			}

			string BranchEndTerminal::print()
			{
				return "]";
			}
		}
	}
}