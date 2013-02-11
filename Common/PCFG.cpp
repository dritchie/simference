#include "PCFG.h"
#include "Distributions.h"

using namespace std;

namespace simference
{
	pair<vector<Grammar::Symbol*>, double> Grammar::Variable::unroll()
	{
		// Accumulate the productions that are actually applicable
		vector<Production*> applicableProds;
		for (Production p : productions())
		{
			if (p.conditionalFunction(this))
				applicableProds.push_back(&p);
		}

		// Figure out their probabilities
		vector<double> probs(applicableProds.size(), 0.0);
		double totalProb = 0.0;
		for (unsigned int i = 0; i < applicableProds.size(); i++)
		{
			double prob = applicableProds[i]->probabilityFunction(this);
			probs[i] = prob;
			totalProb += prob;
		}
		for (unsigned int i = 0; i < probs.size(); i++)
			probs[i] /= totalProb;

		// Sample one proportional to its probability and use it to unroll
		Production* prodToUse = applicableProds[(unsigned int)(MultinomialUnivariateDistribution<double>::Sample(probs))];
		return prodToUse->unrollFunction(this);
	}

	bool Grammar::Derivation::isTerminal()
	{
		for (Symbol* s : symbols)
		{
			if (!s->isTerminal()) return false;
		}
		return true;
	}

	Grammar::Derivation Grammar::Derivation::unroll()
	{
		vector<Symbol*> nextsymbols;
		double probAccum = 1.0;
		for (Symbol* s : symbols)
		{
			// If this is a terminal symbol, just copy it over.
			if (s->isTerminal())
				nextsymbols.push_back(s);
			// Otherwise, unroll the variable and record the probability of doing so.
			else
			{
				Variable* var = (Variable*)s;
				auto nextAndProb = var->unroll();
				probAccum *= nextAndProb.second;
				nextsymbols.insert(nextsymbols.end(), nextAndProb.first.begin(), nextAndProb.first.end());
			}
		}
		return Derivation(nextsymbols, this->probability*probAccum);
	}

	Grammar::DerivationTree Grammar::Sample(const Derivation& axiom)
	{
		DerivationTree dtree;
		dtree.derivations.push_back(axiom);

		while(!dtree.derivations.back().isTerminal())
		{
			dtree.derivations.push_back(dtree.derivations.back().unroll());
		}
		
		return dtree;
	}
}