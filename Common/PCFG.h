#ifndef __PCFG_H
#define __PCFG_H

#include "Distributions.h"
#include <functional>
#include <vector>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <stack>

using namespace simference::Math::Probability;

namespace simference
{
	namespace Grammar
	{
		class Symbol
		{
		public:
			virtual bool isTerminal() = 0;
			virtual void print(std::ostream& outstream) = 0;
			template<class T> bool is() { return dynamic_cast<T*>(this) != NULL; }
			template<class T> T* as() { return dynamic_cast<T*>(this); }
		};

		typedef std::shared_ptr<Symbol> SymbolPtr;

		template<typename RealNum>
		class Terminal : public Symbol
		{
		public:
			typedef typename std::vector<RealNum>::const_iterator ParamIterator;
			bool isTerminal() { return true; }
			virtual RealNum paramLogProb() = 0;
			virtual void getParams(std::vector<RealNum>& p) = 0;
			virtual void setParams(ParamIterator& p) = 0;
		};

		template<typename RealNum, unsigned int nParams>
		class GeneralTerminal : public Terminal<RealNum>
		{
		public:

			GeneralTerminal(Distribution<RealNum>** d)
				: distribs(d)
			{
				for (unsigned int i = 0; i < nParams; i++)
					params[i] = distribs[i]->sample();
			}

			RealNum paramLogProb()
			{
				RealNum lp = 0.0;
				for (unsigned int i = 0; i < nParams; i++)
					lp += distribs[i]->logprob(params[i]);
				return lp;
			}

			void getParams(std::vector<RealNum>& p)
			{
				for (unsigned int i = 0; i < nParams; i++)
					p.push_back(params[i]);
			}

			void setParams(ParamIterator& p)
			{
				for (unsigned int i = 0; i < nParams; i++)
				{
					params[i] = *p;
					p++;
				}
			}

			void print(std::ostream& outstream)
			{
				outstream << name() << "(";
				for (unsigned int i = 0; i < nParams; i++)
					outstream << params[i] << ",";
				outstream << ")";
			}

			virtual char* name() = 0;

			RealNum params[nParams];
			Distribution<RealNum>** distribs;
		};

		template<typename RealNum>
		class String
		{
		public:

			String(const std::vector<SymbolPtr>& syms, RealNum lp)
				: logprob(lp), symbols(syms) {}

			String() : logprob(0.0) {}

			RealNum structureLogProb() { return logprob; }

			RealNum paramLogProb()
			{
				RealNum lp = 0.0;
				for (auto sym : symbols)
				{
					if (sym->isTerminal())
						lp += sym->as<Terminal<RealNum>>()->paramLogProb();
				}
				return lp;
			}

			RealNum totalLogProb() { return structureLogProb() + paramLogProb(); }

			void getParams(std::vector<RealNum>& p)
			{
				for (auto sym : symbols)
				{
					if (sym->isTerminal())
						sym->as<Terminal<RealNum>>()->getParams(p);
				}
			}

			void setParams(const std::vector<RealNum>& p)
			{
				auto it = p.begin();
				for (auto sym : symbols)
				{
					if (sym->isTerminal())
						sym->as<Terminal<RealNum>>()->setParams(it);
				}
			}

			RealNum logprob;
			std::vector<SymbolPtr> symbols;
		};

		// Forward declaration
		template<typename RealNum> class Production;

		template<typename RealNum>
		class Variable : public Symbol
		{
		public:

			bool isTerminal() { return false; }

			String<RealNum> unroll()
			{
				// Accumulate the productions that are actually applicable
				const vector<Production<RealNum>>& prods = productions();
				vector<unsigned int> applicableProds;
				for (unsigned int i = 0; i < prods.size(); i++)
				{
					if (prods[i].conditionalFunction(*this))
						applicableProds.push_back(i);
				}

				// Figure out their probabilities
				vector<RealNum> probs;
				RealNum totalProb = 0.0;
				for (unsigned int i : applicableProds)
				{
					RealNum prob = prods[i].probabilityFunction(*this);
					probs.push_back(prob);
					totalProb += prob;
				}
				for (unsigned int i = 0; i < probs.size(); i++)
					probs[i] /= totalProb;

				// Sample one proportional to its probability and use it to unroll
				unsigned int indexToUse = (unsigned int)(MultinomialDistribution<RealNum>::Sample(probs));
				const Production<RealNum>& prodToUse = prods[applicableProds[indexToUse]];
				RealNum probability = probs[indexToUse];
				auto symbols = prodToUse.unrollFunction(*this);
				return String<RealNum>(symbols, log(probability));
			}

			virtual const std::vector<Production<RealNum>>& productions() = 0;
		};
		
		template<typename RealNum>
		class Production
		{
		public:

			typedef std::function<bool(const Variable<RealNum>&)> ConditionalFunction;
			typedef std::function<RealNum(const Variable<RealNum>&)> ProbabilityFunction;
			typedef std::function<std::vector<SymbolPtr>(const Variable<RealNum>&)> UnrollFunction;

			Production(ConditionalFunction condFunc, ProbabilityFunction probFunc, UnrollFunction unrollFunc)
				: conditionalFunction(condFunc), probabilityFunction(probFunc), unrollFunction(unrollFunc)
			{}

			ConditionalFunction conditionalFunction;
			ProbabilityFunction probabilityFunction;
			UnrollFunction unrollFunction;
		};

		template<typename RealNum>
		class DerivationTree
		{
		public:

			String<RealNum> derivedString()
			{
				// Linearize all terminal symbols (DFS order, insert children in reverse order)
				String<RealNum> derivation;
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
						const String<RealNum>& succ = successorMap[s];
						derivation.logprob += succ.logprob;
						for (auto it = succ.symbols.rbegin(); it != succ.symbols.rend(); it++)
							fringe.push(*it);
					}
				}
				return derivation;
			}

			String<RealNum> roots;
			std::unordered_map<SymbolPtr, String<RealNum> > successorMap;
		};

		template<typename RealNum>
		DerivationTree<RealNum> Sample(const String<RealNum>& axiom)
		{
			DerivationTree<RealNum> dtree;
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
					Variable<RealNum>* var = (Variable<RealNum>*)(s.get());
					auto& newsyms = dtree.successorMap[s] = var->unroll();
					for (auto it = newsyms.symbols.rbegin(); it != newsyms.symbols.rend(); it++)
						fringe.push(*it);
				}
			}		
			return dtree;
		}
	}
}

#endif