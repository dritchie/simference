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
#include <typeinfo>

using namespace simference::Math::Probability;

namespace simference
{
	namespace Grammar
	{
		template <typename RealNum>
		class Symbol
		{
		public:

			Symbol(unsigned int d) : depth(d) {}
			virtual void print(std::ostream& outstream) const = 0;
			virtual void unroll() = 0;
			virtual RealNum logProb() const = 0;
			virtual RealNum recursiveParamLogProb() const = 0;
			virtual RealNum recursiveStructureLogProb() const = 0;
			virtual RealNum recursiveLogProb() const { return recursiveParamLogProb() + recursiveStructureLogProb(); }
			virtual unsigned int numParams() const { return 0; }
			virtual void getParams(std::vector<RealNum>& p) const {}
			virtual void setParams(const ParameterVector<RealNum>& p, unsigned int& pindex) {}
			virtual unsigned int numChildren() const { return 0; }
			virtual const std::vector< std::shared_ptr<Symbol<RealNum>> >& children() const = 0;
			virtual std::shared_ptr<Symbol<RealNum>> deepCopy() const = 0;
			template<class T> bool is() { return dynamic_cast<T*>(this) != NULL; }
			template<class T> T* as() { return dynamic_cast<T*>(this); }

			unsigned int depth;	// in the derivation tree
		};

		template <typename RealNum>
		class SymbolPtr
		{
		public:
			typedef std::shared_ptr<Symbol<RealNum>> type;
		};

		template <typename RealNum>
		class String
		{
		public:
			typedef std::vector<typename SymbolPtr<RealNum>::type> type;
		};

		template<typename RealNum>
		class Terminal : public Symbol<RealNum>
		{
		public:
			Terminal(unsigned int d) : Symbol(d) {}
			void unroll() {}
			RealNum recursiveParamLogProb() const { return logProb(); }
			RealNum recursiveStructureLogProb() const { return 0.0; }
			const typename String<RealNum>::type& children() const { throw "This method should never be called; what's wrong with you!?"; }
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

			RealNum logProb() const
			{
				RealNum lp = 0.0;
				for (unsigned int i = 0; i < nParams; i++)
				{
					lp += distribs[i]->logprob(params[i]);
				}
				return lp;
			}

			unsigned int numParams() const { return nParams; }

			void getParams(std::vector<RealNum>& p) const
			{
				for (unsigned int i = 0; i < nParams; i++)
					p.push_back(params[i]);
			}

			void setParams(const ParameterVector<RealNum>& p, unsigned int& pindex)
			{
				for (unsigned int i = 0; i < nParams; i++, pindex++)
				{
					params[i] = p[pindex];
				}
			}

			virtual GeneralTerminal<RealNum, nParams>* copy() const = 0;

			typename SymbolPtr<RealNum>::type deepCopy() const
			{
				auto gt = copy();
				for (unsigned int i = 0; i < nParams; i++)
					gt->params[i] = params[i];
				return SymbolPtr<RealNum>::type(gt);
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
		class Variable : public Symbol<RealNum>
		{
		public:

			Variable(unsigned int d) : Symbol(d) {}

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
				childSyms = prodToUse.unrollFunction(*this);

				// Recursively unroll all children
				for (auto child : childSyms)
				{
					child->unroll();
				}
			}

			virtual Variable<RealNum>* copy() const = 0;

			typename SymbolPtr<RealNum>::type deepCopy() const
			{
				auto v = copy();
				v->unrolledProduction = unrolledProduction;
				for (auto s : childSyms)
					v->childSyms.push_back(s->deepCopy());
				return SymbolPtr<RealNum>::type(v);
			}

			unsigned int numChildren() const
			{
				return childSyms.size();
			}

			const typename String<RealNum>::type& children() const
			{
				return childSyms;
			}

			RealNum logProb() const
			{
				if (childSyms.size() > 0)
					return log(productions()[unrolledProduction].probabilityFunction(*this));
				else return 0.0;
			}

			RealNum recursiveStructureLogProb() const
			{
				RealNum lp = logProb();
				for (auto c : childSyms)
					lp += c->recursiveStructureLogProb();
				return lp;
			}

			RealNum recursiveParamLogProb() const
			{
				RealNum lp = 0.0;
				for (auto c : childSyms)
					lp += c->recursiveParamLogProb();
				return lp;
			}

			// Overriding recursiveLogProb in this way allows us to use one tree traversal instead of two.
			RealNum recursiveLogProb()
			{
				RealNum lp = logProb();
				for (auto c : childSyms)
					lp += c->recursiveLogProb();
				return lp;
			}

			virtual const std::vector<Production<RealNum>>& productions() const = 0;

			typename String<RealNum>::type childSyms;
			unsigned int unrolledProduction;
		};

		template<typename RealNum>
		class Production
		{
		public:

			typedef std::function<bool(const Variable<RealNum>&)> ConditionalFunction;
			typedef std::function<RealNum(const Variable<RealNum>&)> ProbabilityFunction;
			typedef std::function<std::vector<typename SymbolPtr<RealNum>::type>(const Variable<RealNum>&)> UnrollFunction;

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

			DerivationTree(const typename String<RealNum>::type& axiom)
				: roots(axiom)
			{
				for (auto sym : roots)
					sym->unroll();
				computeDerivation();
			}

			DerivationTree(const DerivationTree<RealNum>& dt)
			{
				provenance = dt.provenance;
				for (auto r : dt.roots)
					roots.push_back(r->deepCopy());
				computeDerivation();

				// We don't update provenance here...that only happens in LARJ proposals
			}

			bool structurallyEquivalentTo(StructurePtr other)
			{
				std::shared_ptr<DerivationTree<RealNum>> dt = dynamic_pointer_cast<DerivationTree<RealNum>>(other);
				if (!dt) return false;

				// Linearize variables. Check that they are all of the same type
				// and made the same structural choices.
				typename String<RealNum>::type vars1;
				typename String<RealNum>::type vars2;
				this->variables(vars1);
				dt->variables(vars2);
				for (unsigned int i = 0; i < vars1.size(); i++)
				{
					auto v1 = static_pointer_cast<Variable<RealNum>>(vars1[i]);
					auto v2 = static_pointer_cast<Variable<RealNum>>(vars2[i]);
					if (typeid(v1) != typeid(v2) || v1->unrolledProduction != v2->unrolledProduction)
						return false;
				}
				return true;
			}

			void reroll(Variable<RealNum>& v)
			{
				v.unroll();
				computeDerivation();
			}

			void printFullTree(std::ostream& out) const
			{
				std::stack<typename SymbolPtr<RealNum>::type> fringe;
				for (auto it = roots.rbegin(); it != roots.rend(); it++)
					fringe.push(*it);
				while (!fringe.empty())
				{
					typename SymbolPtr<RealNum>::type s = fringe.top();
					fringe.pop();
					for (unsigned int i = 0; i < s->depth; i++)
						out << "  ";
					s->print(out);
					out << std::endl;
					if (s->numChildren() > 0)
					{
						const auto& children = s->children();
						for (auto it = children.rbegin(); it != children.rend(); it++)
							fringe.push(*it);
					}
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
				for (auto sym : roots)
					lp += sym->recursiveStructureLogProb();
				return lp;
			}

			RealNum paramLogProb() const
			{
				RealNum lp = 0.0;
				for (auto sym : roots)
					lp += sym->recursiveParamLogProb();
				return lp;
			}

			RealNum logProb() const
			{
				RealNum lp = 0.0;
				for (auto sym : roots)
					lp += sym->recursiveLogProb();
				return lp;
			}

			unsigned int numParams() const
			{
				unsigned int n = 0;
				for (auto sym : derivation)
					n += sym->numParams();
				return n;
			}

			void getParams(std::vector<RealNum>& p) const
			{
				for (auto sym : derivation)
					sym->getParams(p);
			}

			void setParams(const ParameterVector<RealNum>& p)
			{
				unsigned int pindex = 0;
				for (auto sym : derivation)
					sym->setParams(p, pindex);
			}

			void computeDerivation()
			{
				derivation.clear();
				stack<typename SymbolPtr<RealNum>::type> fringe;
				for (auto it = roots.rbegin(); it != roots.rend(); it++)
					fringe.push(*it);
				while (!fringe.empty())
				{
					auto s = fringe.top();
					fringe.pop();
					if (s->numChildren() == 0)
						derivation.push_back(s);
					else
					{
						const auto& children = s->children();
						for (auto it = children.crbegin(); it != children.crend(); it++)
							fringe.push(*it);
					}
				}
			}

			void variables(typename String<RealNum>::type& vars) const
			{
				stack<typename SymbolPtr<RealNum>::type> fringe;
				for (auto it = roots.rbegin(); it != roots.rend(); it++)
					fringe.push(*it);
				while (!fringe.empty())
				{
					auto s = fringe.top();
					fringe.pop();
					if (s->is<Variable<RealNum>>())
						vars.push_back(s);
					if (s->numChildren() > 0)
					{
						const auto& children = s->children();
						for (auto it = children.crbegin(); it != children.crend(); it++)
							fringe.push(*it);
					}
				}
			}

			typename String<RealNum>::type roots;
			typename String<RealNum>::type derivation;

			class Provenance
			{
			public:
				Provenance() : modifiedFrom(NULL), oldSubtreeRoot(NULL), newSubtreeRoot(NULL) {}
				std::shared_ptr<DerivationTree<RealNum>> modifiedFrom;
				// We could at some point generalize this to be an arbitrary number of old/new subtree pairs?
				typename SymbolPtr<RealNum>::type oldSubtreeRoot;
				typename SymbolPtr<RealNum>::type newSubtreeRoot;
			};
			Provenance provenance;
		};
	}
}

#endif