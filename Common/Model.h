#ifndef __MODEL_H
#define __MODEL_H

#include <stan/model/prob_grad_ad.hpp>
#include <functional>

namespace simference
{
	class Structure
	{
	public:
		virtual void getParams(std::vector<double>& params) = 0;
		virtual void setParams(const std::vector<double>& params) = 0;
		virtual unsigned int numParams() = 0;
	};

	typedef std::shared_ptr<Structure> StructurePtr;

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
			virtual stan::agrad::var log_prob(std::vector<stan::agrad::var>& params_r) = 0;
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

		class FactorTemplate
		{
		public:
			virtual void unroll(StructurePtr s, std::vector<FactorPtr>& factors) = 0;
			// Default behavior is inefficient: the above unroll is invoked on sOld and sNew separately.
			virtual void unroll(StructurePtr sOld, StructurePtr sNew,
				std::vector<FactorPtr>& fOld, std::vector<FactorPtr>& fNew, std::vector<FactorPtr>& fShared);
		};

		typedef std::shared_ptr<FactorTemplate> FactorTemplatePtr;

		class FactorTemplateModel
		{
		public:
			FactorTemplateModel() {}
			FactorTemplateModel(const std::vector<FactorTemplatePtr>& ts)
				: templates(ts) {}
			void addTemplate(FactorTemplatePtr t);
			void removeTemplate(FactorTemplatePtr t);
			void unroll(StructurePtr s, std::vector<FactorPtr>& factors);
			void unroll(StructurePtr sOld, StructurePtr sNew,
				std::vector<FactorPtr>& fOld, std::vector<FactorPtr>& fNew, std::vector<FactorPtr>& fShared);
		private:
			std::vector<FactorTemplatePtr> templates;
		};

		typedef std::shared_ptr<FactorTemplateModel> FactorTemplateModelPtr;

		// Weights are not required to be normalized.
		class MixtureModel : public Model
		{
		public:
			MixtureModel(const std::vector<ModelPtr>& ms, const std::vector<double>& ws);
			MixtureModel(const std::vector<ModelPtr>& ms);
			stan::agrad::var log_prob(std::vector<stan::agrad::var>& params_r);
			std::vector<double>& getWeights() { return weights; }

		private:
			std::vector<ModelPtr> models;
			std::vector<double> weights;
		};
	}
}

#endif