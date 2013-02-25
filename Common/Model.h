#ifndef __MODEL_H
#define __MODEL_H

#include <stan/model/prob_grad_ad.hpp>
#include <functional>

namespace simference
{
	namespace Models
	{
		class Model : public stan::model::prob_grad_ad
		{
		public:
			Model(unsigned int nParams) : prob_grad_ad(nParams) {}
			virtual stan::agrad::var log_prob(std::vector<stan::agrad::var>& params_r) = 0; 
			stan::agrad::var log_prob(std::vector<stan::agrad::var>& params_r, 
				std::vector<int>& params_i,
				std::ostream* output_stream = 0)
			{
				return log_prob(params_r);
			}
		};

		typedef std::shared_ptr<Model> ModelPtr;

		class Factor
		{
		public:
			Factor(unsigned int pOffset) : paramOffset(pOffset) {}
			unsigned int parameterOffset() { return paramOffset; }
			virtual stan::agrad::var log_prob(std::vector<stan::agrad::var>::const_iterator& pit) = 0;
		private:
			unsigned int paramOffset;
		};

		typedef std::shared_ptr<Factor> FactorPtr;

		class FactorModel : public Model
		{
		public:
			FactorModel(unsigned int nParams) : Model(nParams) {}
			FactorModel(unsigned int nParams, const std::vector<FactorPtr>& fs)
				: Model(nParams), factors(fs) {}
			void addFactor(FactorPtr f);
			void removeFactor(FactorPtr f);
			stan::agrad::var log_prob(std::vector<stan::agrad::var>& params_r); 

		protected:
			std::vector<FactorPtr> factors;
		};

		template <class Structure>
		class FactorTemplate
		{
		public:
			virtual void unroll(const Structure& s, std::vector<FactorPtr>& factors) = 0;
		};

		// Because MSVC doesn't yet support alias templates
		template <class Structure>
		class FactorTemplatePtr
		{
		public:
			typedef std::shared_ptr<FactorTemplate<Structure>> type;
		};

		template <class Structure>
		class FactorTemplateModel
		{
		public:
			typedef typename FactorTemplatePtr<Structure>::type FacTempPtr;
			FactorTemplateModel() {}
			FactorTemplateModel(const std::vector<FacTempPtr>& ts)
				: templates(ts) {}
			void addTemplate(FacTempPtr t)
			{
				templates.push_back(t);
			}
			void removeTemplate(FacTempPtr t)
			{
				for (auto it = templates.begin(); it != templates.end(); it++)
				{
					if (*it == t)
					{
						templates.erase(it);
						return;
					}
				}
			}
			void unroll(const Structure& s, std::vector<FactorPtr>& factors)
			{
				for (auto t : templates)
					t->unroll(s, factors);
			}
		private:
			std::vector<FacTempPtr> templates;
		};

		// Weights are not required to be normalized.
		class MixtureModel : public Model
		{
		public:
			MixtureModel(const std::vector<ModelPtr>& ms, const std::vector<double>& ws);
			stan::agrad::var log_prob(std::vector<stan::agrad::var>& params_r);
			std::vector<double>& getWeights() { return weights; }

		private:
			std::vector<ModelPtr> models;
			std::vector<double> weights;
		};
	}
}

#endif