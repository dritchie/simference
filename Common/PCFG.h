#ifndef __PCFG_H
#define __PCFG_H

#include <functional>
#include <vector>
#include <utility>

namespace simference
{
	class Grammar
	{
		class Variable;
		class Symbol;

		typedef std::function<bool(Variable*)> ProductionConditionalFunction;
		typedef std::function<double(Variable*)> ProductionProbabilityFunction;
		typedef std::function<std::pair<std::vector<Symbol*>,double>(Variable*)> ProductionUnrollFunction;

		class Production
		{
		public:
			Production(ProductionConditionalFunction condFunc,
				ProductionProbabilityFunction probFunc,
				ProductionUnrollFunction unrollFunc)
				:
			conditionalFunction(condFunc),
				probabilityFunction(probFunc),
				unrollFunction(unrollFunc)
			{}

			ProductionConditionalFunction conditionalFunction;
			ProductionProbabilityFunction probabilityFunction;
			ProductionUnrollFunction unrollFunction;
		};

		typedef std::vector<Production> ProductionList;

		class Symbol
		{
		public:
			virtual bool isTerminal() = 0;
		};

		class Variable : public Symbol
		{
		public:
			bool isTerminal() { return false; }
			std::pair<std::vector<Symbol*>, double> unroll();
			virtual const ProductionList& productions() = 0;
		};

		class Terminal : public Symbol
		{
		public:
			bool isTerminal() { return true; }
		};

		class Derivation
		{
		public:
			Derivation() : probability(1.0) {}
			Derivation(const std::vector<Symbol*> syms, double prob)
				: symbols(syms), probability(prob) {}
			Derivation unroll();
			std::vector<Symbol*> symbols;
			bool isTerminal();
			double probability;
		};

		class DerivationTree
		{
		public:
			std::vector<Derivation> derivations;
		};

		static DerivationTree Sample(const Derivation& axiom);
	};
}

#endif