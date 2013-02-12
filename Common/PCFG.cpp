#include "PCFG.h"
#include "Distributions.h"
#include <stack>

using namespace std;

namespace simference
{
	namespace Grammar
	{
		String Variable::unroll()
		{
			// Accumulate the productions that are actually applicable
			const vector<Production>& prods = productions();
			vector<unsigned int> applicableProds;
			for (unsigned int i = 0; i < prods.size(); i++)
			{
				if (prods[i].conditionalFunction(*this))
					applicableProds.push_back(i);
			}

			// Figure out their probabilities
			vector<double> probs;
			double totalProb = 0.0;
			for (unsigned int i : applicableProds)
			{
				double prob = prods[i].probabilityFunction(*this);
				probs.push_back(prob);
				totalProb += prob;
			}
			for (unsigned int i = 0; i < probs.size(); i++)
				probs[i] /= totalProb;

			// Sample one proportional to its probability and use it to unroll
			unsigned int indexToUse = (unsigned int)(MultinomialDistribution<double>::Sample(probs));
			const Production& prodToUse = prods[applicableProds[indexToUse]];
			double probability = probs[indexToUse];
			auto succData = prodToUse.unrollFunction(*this);
			succData.logprob += log(probability);
			return succData;
		}

		String DerivationTree::derivedString()
		{
			// Linearize all terminal symbols (DFS order, insert children in reverse order)
			String derivation;
			stack<SymbolPtr> fringe;
			for (auto it = roots.symbols.rbegin(); it != roots.symbols.rend(); it++)
				fringe.push(*it);
			while (!fringe.empty())
			{
				SymbolPtr s = fringe.top();
				fringe.pop();
				if (s->isTerminal())
					derivation.symbols.push_back(s);
				else
				{
					const String& succ = successorMap[s];
					derivation.logprob += succ.logprob;
					for (auto it = succ.symbols.rbegin(); it != succ.symbols.rend(); it++)
						fringe.push(*it);
				}
			}
			return derivation;
		}

		DerivationTree Sample(const String& axiom)
		{
			DerivationTree dtree;
			dtree.roots = axiom;
			dtree.roots.logprob = 0.0;

			stack<SymbolPtr> fringe;
			for (auto it = dtree.roots.symbols.rbegin(); it != dtree.roots.symbols.rend(); it++)
				fringe.push(*it);
			while (!fringe.empty())
			{
				SymbolPtr s = fringe.top();
				fringe.pop();
				if (!s->isTerminal())
				{
					Variable* var = (Variable*)(s.get());
					dtree.successorMap[s] = var->unroll();
				}
			}		
			return dtree;
		}
	}
}