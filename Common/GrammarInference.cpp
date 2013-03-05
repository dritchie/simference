#include "GrammarInference.h"
#include <stack>

using namespace std;
using namespace simference::Grammar;
using namespace stan::agrad;

namespace simference
{
	namespace Models
	{
		void GrammarFactorTemplate::unroll(StructurePtr s, std::vector<FactorPtr>& factors) const
		{
			unordered_set<SymbolPtr<var>::type> exclude;
			factors.push_back(FactorPtr(
				new GrammarFactorTemplate::Factor(s, static_cast<DerivationTree<var>*>(s.get())->roots, exclude)));
		}

		void GrammarFactorTemplate::unroll(StructurePtr sOld, StructurePtr sNew,
			std::vector<FactorPtr>& fOld, std::vector<FactorPtr>& fNew, std::vector<FactorPtr>& fShared) const
		{
			auto dtOld = static_cast<DerivationTree<var>*>(sOld.get());
			auto dtNew = static_cast<DerivationTree<var>*>(sNew.get());
			assert(dtNew->provenance.modifiedFrom.get() == dtOld);

			String<var>::type root;
			unordered_set<SymbolPtr<var>::type> exclude;
			root.push_back(dtNew->provenance.oldSubtreeRoot);
			fOld.push_back(FactorPtr(new GrammarFactorTemplate::Factor(sOld, root, exclude)));
			root.clear(); root.push_back(dtNew->provenance.newSubtreeRoot);
			fNew.push_back(FactorPtr(new GrammarFactorTemplate::Factor(sNew, root, exclude)));
			exclude.insert(dtNew->provenance.oldSubtreeRoot);
			fShared.push_back(FactorPtr(new GrammarFactorTemplate::Factor(sOld, dtOld->roots, exclude)));
		}

		GrammarFactorTemplate::Factor::Factor(StructurePtr dtree,
					   const String<var>::type & roots,
					   const unordered_set<SymbolPtr<var>::type>& exclude)
					   : simference::Models::Factor(dtree)
		{
			// Extract all descendants of 'roots,' except descendants of those in the 'exclude' set
			stack<SymbolPtr<var>::type> fringe;
			for (auto s : roots) fringe.push(s);
			while (!fringe.empty())
			{
				auto s = fringe.top();
				fringe.pop();
				if (exclude.count(s) == 0)
				{
					syms.push_back(s);
					if (s->numChildren() > 0)
					{
						const auto& children = s->children();
						for (auto c : children)
							fringe.push(c);
					}
				}
			}
		}

		var GrammarFactorTemplate::Factor::log_prob(const ParameterVector<var>& params)
		{
			var lp = 0.0;

			// Set the parameters of the derivation
			static_cast<DerivationTree<var>*>(structUnrolledFrom.get())->setParams(params);

			// Accumulate probabilities for all the symbols we've been told to care about
			for (auto s : syms)
				lp += s->logProb();

			return lp;
		}
	}

	namespace Samplers
	{
		StructurePtr GrammarJumpSampler::jumpProposal()
		{
			// Pick a random nonterminal and reroll it.
			// (Don't forget to store the correct information in the 'provenance' field)
			// NOTE: This requires deep copy!
			// We also precompute and store the forward/reverse proposal probabilities.

			auto variableUnrollProbs = [](const String<var>::type& vars, vector<double>& probs)
			{
				unsigned int maxdepth = (*max_element(vars.begin(), vars.end(), [](const SymbolPtr<var>::type& s1, const SymbolPtr<var>::type& s2) { return s1->depth < s2->depth; }))->depth;
				double z = 0.0;
				for (auto s : vars)
				{
					// 2.0 is a crude estimate of branching factor which may not hold at all...
					// TODO: make this a virtual method that subclasses for different grammars would overload??
					double prob = pow(2.0, maxdepth - s->depth);
					z += prob;
					probs.push_back(prob);
				}
				for (unsigned int i = 0; i < probs.size(); i++)
					probs[i] /= z;	// normalize
			};

			// Deep copy
			auto currdt = static_cast<DerivationTree<var>*>(currentStruct.get());
			auto newdt = new DerivationTree<var>(*currdt);
			
			// Decide which variable to reroll
			vector<SymbolPtr<var>::type> currvars, newvars;
			currdt->variables(currvars);
			newdt->variables(newvars);
			vector<double> probabilities;
			variableUnrollProbs(currvars, probabilities);
			unsigned int whichVar = MultinomialDistribution<double>::Sample(probabilities);

			// Re-roll variable in the newly copied tree
			newdt->reroll(*newvars[whichVar]->as<Variable<var>>());

			// Record provenance
			newdt->provenance.modifiedFrom = std::static_pointer_cast<DerivationTree<var>>(currentStruct);
			newdt->provenance.oldSubtreeRoot = currvars[whichVar];
			newdt->provenance.newSubtreeRoot = newvars[whichVar];

			// Record the forward and reverse probabilities (as well as the structures)
			lastStructJumpedFrom = currentStruct;
			lastStructJumpedTo = StructurePtr(newdt);
			lastJumpForwardLp = MultinomialDistribution<double>::Prob(whichVar, probabilities) + newvars[whichVar]->recursiveStructureLogProb().val();
			newvars.clear();
			probabilities.clear();
			newdt->variables(newvars);
			variableUnrollProbs(newvars, probabilities);
			lastJumpReverseLp = MultinomialDistribution<double>::Prob(whichVar, probabilities) + currvars[whichVar]->recursiveStructureLogProb().val();
			
			return lastStructJumpedTo;
		}

		void GrammarJumpSampler::dimensionMatch(StructurePtr sFrom, const std::vector<double>& pFrom,
			StructurePtr sTo, std::vector<double>& pTo, DimensionMatchMap& matching)
		{
			// Verify via provenance that sTo came from sFrom
			// Check which structure is bigger (if the same size, arbitrarily go with the direction old -> new)
			// Traverse sOld, linearizing the parameters, but skip the replaced node. Remember the index in the list where the skip happened.
			// Linearize the old and new subtrees; slap the old params on top of the new ones.
			// Build the map (should be pretty simple)

			// Unboxing and checks
			auto dtFrom = static_cast<DerivationTree<var>*>(sFrom.get());
			auto dtTo = static_cast<DerivationTree<var>*>(sTo.get());
			assert(dtTo->provenance.modifiedFrom.get() == dtFrom);

			auto linearizeParams = [](const String<var>::type& roots, SymbolPtr<var>::type skip, vector<var>& outParams, unsigned int& skipPoint)
			{
				stack<SymbolPtr<var>::type> fringe;
				for (auto it = roots.rbegin(); it != roots.rend(); it++)
					fringe.push(*it);
				while (!fringe.empty())
				{
					auto s = fringe.top();
					fringe.pop();
					if (s.get() == skip.get()) 
					{
						skipPoint = outParams.size();
						continue;
					}
					s->getParams(outParams);
					if (s->numChildren() > 0)
					{
						auto children = s->children();
						for (auto it = children.rbegin(); it != children.rend(); it++)
							fringe.push(*it);
					}
				}
			};

			// Linearize parameters, but skip the old subtree
			vector<var> params;
			unsigned int skipPoint;
			auto skipVar = dtTo->provenance.oldSubtreeRoot;
			linearizeParams(dtFrom->roots, skipVar, params, skipPoint);

			// Linearize both the old and new subtrees
			vector<var> oldTreeParams, newTreeParams;
			unsigned int dummySkipPoint;
			String<var>::type roots;
			roots.push_back(dtTo->provenance.oldSubtreeRoot);
			linearizeParams(roots, SymbolPtr<var>::type(NULL), oldTreeParams, dummySkipPoint);
			roots.clear(); roots.push_back(dtTo->provenance.newSubtreeRoot);
			linearizeParams(roots, SymbolPtr<var>::type(NULL), newTreeParams, dummySkipPoint);

			// Unify the old+new subtree params
			vector<var> unifiedSubtreeParams;
			if (sFrom->numParams() <= sTo->numParams())
			{
				// We have more more new params than old, so concatenate old with the extra new ones.
				unifiedSubtreeParams.insert(unifiedSubtreeParams.end(), oldTreeParams.begin(), oldTreeParams.end());
				unifiedSubtreeParams.insert(unifiedSubtreeParams.end(), newTreeParams.begin() + oldTreeParams.size(), newTreeParams.end());
			}
			else
			{
				// We have more old params than new, so the new param list is literally just the old one
				unifiedSubtreeParams = oldTreeParams;
			}

			// Insert the unified subtree param list into the overall param list
			// (Meanwhile, convert to doubles)
			for (unsigned int i = 0; i < skipPoint; i++)
				pTo.push_back(params[i].val());
			for (auto p : unifiedSubtreeParams)
				pTo.push_back(p.val());
			for (unsigned int i = skipPoint; i < params.size(); i++)
				pTo.push_back(params[i].val());

			// Build the parameter index map
			if (sFrom->numParams() <= sTo->numParams())
			{
				matching.direction = DimensionMatchMap::OldToNew;
				for (unsigned int i = 0; i < skipPoint + oldTreeParams.size(); i++)
					matching.paramIndexMap.push_back(i);
				for (unsigned int i = skipPoint + unifiedSubtreeParams.size(); i < pTo.size(); i++)
					matching.paramIndexMap.push_back(i);
			}
			else
			{
				matching.direction = DimensionMatchMap::NewToOld;
				for (unsigned int i = 0; i < skipPoint + newTreeParams.size(); i++)
					matching.paramIndexMap.push_back(i);
				for (unsigned int i = skipPoint + unifiedSubtreeParams.size(); i < pTo.size(); i++)
					matching.paramIndexMap.push_back(i);
			}
		}

		double GrammarJumpSampler::logProposalProbability(StructurePtr sFrom, const std::vector<double>& pFrom,
			StructurePtr sTo, const std::vector<double>& pTo)
		{
			// Just look up one of the probabilities we precomputed in 'jumpProposal'
			if (sFrom.get() == lastStructJumpedFrom.get())
				return lastJumpForwardLp;
			else if (sFrom.get() == lastStructJumpedTo.get())
				return lastJumpReverseLp;
			else throw "Sorry, I've never seen this structure before!";
		}
	}
}