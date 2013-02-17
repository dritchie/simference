#ifndef __PCFG_H
#define __PCFG_H

#include "Distributions.h"
#include <functional>
#include <vector>
#include <memory>
#include <unordered_map>
#include <iostream>

namespace simference
{
	namespace Grammar
	{
		typedef simference::Math::Probability::Distribution<double>* DistribPtr;
		class Symbol
		{
		public:
			virtual bool isTerminal() = 0;
			virtual void print(std::ostream& outstream) = 0;
			template<class T> bool is() { return dynamic_cast<T*>(this) != NULL; }
			template<class T> T* as() { return dynamic_cast<T*>(this); }
		};

		typedef std::shared_ptr<Symbol> SymbolPtr;

		class String
		{
		public:
			String(const std::vector<SymbolPtr>& syms, double lp)
				: logprob(lp), symbols(syms) {}
			String() : logprob(0.0) {}
			double structureLogProb() { return logprob; }
			double paramLogProb();
			double totalLogProb() { return structureLogProb() + paramLogProb(); }
			void getParams(std::vector<double>& p);
			void setParams(const std::vector<double>& p);
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
			virtual double paramLogProb() = 0;
			virtual void getParams(std::vector<double>& p) = 0;
			virtual void setParams(std::vector<double>::const_iterator& p) = 0;
		};

		template<unsigned int nParams>
		class GeneralTerminal : public Terminal
		{
		public:
			GeneralTerminal(DistribPtr* d)
				: distribs(d)
			{
				for (unsigned int i = 0; i < nParams; i++)
					params[i] = distribs[i]->sample();
			}
			double paramLogProb()
			{
				double lp = 0.0;
				for (unsigned int i = 0; i < nParams; i++)
					lp += distribs[i]->logprob(params[i]);
				return lp;
			}
			void getParams(std::vector<double>& p)
			{
				for (unsigned int i = 0; i < nParams; i++)
					p.push_back(params[i]);
			}
			void setParams(std::vector<double>::const_iterator& p)
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
			double params[nParams];
			DistribPtr* distribs;
		};

		typedef std::function<bool(const Variable&)> ProductionConditionalFunction;
		typedef std::function<double(const Variable&)> ProductionProbabilityFunction;
		typedef std::function<std::vector<SymbolPtr>(const Variable&)> ProductionUnrollFunction;

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