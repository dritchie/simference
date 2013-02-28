#ifndef __PCFG_H
#define __PCFG_H

#include "Distributions.h"
#include "Model.h"
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

			Symbol(unsigned int d) : depth(d) {}
			virtual bool isTerminal() const = 0;
			virtual void print(std::ostream& outstream) const = 0;
			virtual void unroll() = 0;
			template<class T> bool is() { return dynamic_cast<T*>(this) != NULL; }
			template<class T> T* as() { return dynamic_cast<T*>(this); }

			unsigned int depth;	// in the derivation tree
		};

		typedef std::shared_ptr<Symbol> SymbolPtr;

		typedef std::vector<SymbolPtr> String;

		template<typename RealNum>
		class Terminal : public Symbol
		{
		public:
			typedef typename std::vector<RealNum>::const_iterator ParamIterator;
			Terminal(unsigned int d) : Symbol(d) {}
			bool isTerminal() const { return true; }
			void unroll() {}
			virtual RealNum paramLogProb() const = 0;
			virtual unsigned int numParams() const = 0;
			virtual void getParams(std::vector<RealNum>& p) const = 0;
			virtual void setParams(ParamIterator& p) = 0;
		};

		template<typename RealNum, unsigned int nParams>
		class GeneralTerminal : public Terminal<RealNum>
		{
		public:

			GeneralTerminal(unsigned int d, Distribution<RealNum>** dis)
				: Terminal(d), distribs(dis)
			{
				for (unsigned int i = 0; i < nParams; i++)
					params[i] = distribs[i]->sample();
			}

			RealNum paramLogProb() const
			{
				RealNum lp = 0.0;
				for (unsigned int i = 0; i < nParams; i++)
					lp += distribs[i]->logprob(params[i]);
				return lp;
			}

			unsigned int numParams() const { return nParams; }

			void getParams(std::vector<RealNum>& p) const
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

			void print(std::ostream& outstream) const
			{
				outstream << name() << "(";
				for (unsigned int i = 0; i < nParams; i++)
					outstream << params[i] << ",";
				outstream << ")";
			}

			virtual char* name() const = 0;

			RealNum params[nParams];
			Distribution<RealNum>** distribs;
		};


		// Forward declaration
		template<typename RealNum> class Production;

		template<typename RealNum>
		class Variable : public Symbol
		{
		public:

			Variable(unsigned int d) : Symbol(d) {}

			bool isTerminal() const { return false; }

			void unroll()
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
				unrolledProduction = MultinomialDistribution<RealNum>::Sample(probs);
				const Production<RealNum>& prodToUse = prods[applicableProds[unrolledProduction]];
				children = prodToUse.unrollFunction(*this);

				// Recursively unroll all children
				for (auto child : children)
				{
					child->unroll();
				}
			}

			virtual const std::vector<Production<RealNum>>& productions() = 0;

			String children;
			unsigned int unrolledProduction;
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
		class DerivationTree : public Structure
		{
		public:

			static DerivationTree<RealNum> Derive(const String& axiom)
			{
				DerivationTree<RealNum> dtree;
				dtree.roots = axiom;
				for (auto sym : dtree.roots)
					sym->unroll();
				dtree.computeDerivation();
				return dtree;
			}

			void printFullTree(std::ostream& out) const
			{
				std::stack<SymbolPtr> fringe;
				for (auto it = roots.rbegin(); it != roots.rend(); it++)
					fringe.push(*it);
				while (!fringe.empty())
				{
					SymbolPtr s  = fringe.top();
					fringe.pop();
					for (unsigned int i = 0; i < s->depth; i++)
						out << "  ";
					sym->print(out);
					out << std::endl;
				}
			}

			void printDerivation(std::ostream& out) const
			{
				for (auto sym : derivation)
					sym->print(out);
				out << std::endl;
			}

			RealNum structureLogProb() const
			{
				RealNum lp = 0.0;
				for (auto sym : derivation)
				{
					if (!sym->isTerminal())
					{
						Variable<RealNum>* v = (Variable<RealNum>*)sym->as<Variable<RealNum>>();
						lp += v->productions()[v->unrolledProduction].probabilityFunction(*v);
					}
				}
			}

			RealNum paramLogProb() const
			{
				RealNum lp = 0.0;
				for (auto sym : derivation)
				{
					if (sym->isTerminal())
						lp += sym->as<Terminal<RealNum>>()->paramLogProb();
				}
				return lp;
			}

			RealNum totalLogProb() const { return structureLogProb() + paramLogProb(); }

			unsigned int numParams() const
			{
				unsigned int n = 0;
				for (auto sym : derivation)
				{
					if (sym->isTerminal())
						n += sym->as<Terminal<RealNum>>()->numParams();
				}
				return n;
			}

			void getParams(std::vector<RealNum>& p) const
			{
				for (auto sym : derivation)
				{
					if (sym->isTerminal())
						sym->as<Terminal<RealNum>>()->getParams(p);
				}
			}

			void setParams(const std::vector<RealNum>& p)
			{
				auto it = p.begin();
				for (auto sym : derivation)
				{
					if (sym->isTerminal())
						sym->as<Terminal<RealNum>>()->setParams(it);
				}
			}


			void computeDerivation()
			{
				derivation.clear();
				stack<SymbolPtr> fringe;
				for (auto it = roots.rbegin(); it != roots.rend(); it++)
					fringe.push(*it);
				while (!fringe.empty())
				{
					SymbolPtr s = fringe.top();
					fringe.pop();
					if (s->isTerminal())
						derivation.push_back(s);
					else
					{
						Variable<RealNum>* v = (Variable<RealNum>*)s->as<Variable<RealNum>>();
						for (auto it = v->children.rbegin(); it != v->children.rend(); it++)
							fringe.push(*it);
					}
				}
			}

			String roots;
			String derivation;
		};
	}
}

#endif