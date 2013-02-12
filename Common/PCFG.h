#ifndef __PCFG_H
#define __PCFG_H

#include <functional>
#include <vector>
#include <memory>
#include <unordered_map>

namespace simference
{
	namespace Grammar
	{
		class Symbol
		{
		public:
			virtual bool isTerminal() = 0;
			virtual std::string print() = 0;
		};

		typedef std::shared_ptr<Symbol> SymbolPtr;

		class String
		{
		public:
			String(const std::vector<SymbolPtr>& syms, double lp)
				: logprob(lp), symbols(syms) {}
			String() : logprob(0.0) {}
			double logprob;
			std::vector<SymbolPtr> symbols;
		};

		class Production;
		class Variable : public Symbol
		{
		public:
			bool isTerminal() { return false; }
			String unroll();
			virtual const std::vector<Production>& productions() = 0;
		};

		class Terminal : public Symbol
		{
		public:
			bool isTerminal() { return true; }
		};

		typedef std::function<bool(const Variable&)> ProductionConditionalFunction;
		typedef std::function<double(const Variable&)> ProductionProbabilityFunction;
		typedef std::function<String(const Variable&)> ProductionUnrollFunction;

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

		class DerivationTree
		{
		public:
			String derivedString();
			String roots;
			std::unordered_map<SymbolPtr, String> successorMap;
		};

		DerivationTree Sample(const String& axiom);
	}
}

#endif