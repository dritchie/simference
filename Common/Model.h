#ifndef __MODEL_H
#define __MODEL_H

#include <stan/model/prob_grad_ad.hpp>
#include <functional>

namespace simference
{
	class Structure
	{
	public:
		virtual void getParams(std::vector<double>& params) const = 0;
		virtual void setParams(const std::vector<double>& params) = 0;
		virtual unsigned int numParams() const = 0;
	};

	class DimensionMatchMap
	{
	public:
		std::vector<unsigned int> paramIndexMap;
		enum Direction
		{
			OldToNew = 0,
			NewToOld
		};
		Direction direction;
	};

	typedef std::shared_ptr<Structure> StructurePtr;

	namespace Models
	{
		class Model : public stan::model::prob_grad_ad
		{
		public:
			Model(unsigned int nParams) : prob_grad_ad(nParams) {}
			virtual stan::agrad::var log_prob(const std::vector<stan::agrad::var>& params_r) const = 0; 
			stan::agrad::var log_prob(std::vector<stan::agrad::var>& params_r, 
				std::vector<int>& params_i,
				std::ostream* output_stream = 0)
			{
				return log_prob(params_r);
			}
			double log_prob(std::vector<double>& params_r)
			{
				return stan::model::prob_grad_ad::log_prob(params_r, dummy);
			};
		private:
			vector<int> dummy;
		};

		typedef std::shared_ptr<Model> ModelPtr;

		class FactorParameters
		{
		public:
			FactorParameters(const std::vector<stan::agrad::var>& p)
				: params(p) {}
			virtual const stan::agrad::var& operator[] (unsigned int i) const;
			virtual size_t size() const;
		protected:
			const std::vector<stan::agrad::var>& params;
		};

		class DimensionMatchedFactorParameters : public FactorParameters
		{
		public:
			DimensionMatchedFactorParameters(const std::vector<stan::agrad::var>& p,
				const std::vector<unsigned int>& imap)
				: FactorParameters(p), indexMap(imap) {}
			const stan::agrad::var& operator[] (unsigned int i) const;
			size_t size() const;
		protected:
			const std::vector<unsigned int>& indexMap;
		};

		typedef std::shared_ptr<FactorParameters> FactorParametersPtr;

		class Factor
		{
		public:
			Factor(StructurePtr s) : structUnrolledFrom(s) {}
			virtual stan::agrad::var log_prob(FactorParametersPtr params) const = 0;

		protected:
			friend class FactorModel;
			StructurePtr structUnrolledFrom;
		};

		typedef std::shared_ptr<Factor> FactorPtr;

		class FactorModel : public Model
		{
		public:
			FactorModel(StructurePtr s, unsigned int nParams, const std::vector<FactorPtr>& fs);
			stan::agrad::var log_prob(const std::vector<stan::agrad::var>& params_r) const; 

		protected:
			virtual FactorParametersPtr wrapParameters(const std::vector<stan::agrad::var>& params_r) const;
			StructurePtr structUnrolledFrom;
			std::vector<FactorPtr> factors;
		};

		class DimensionMatchedFactorModel : public FactorModel
		{
		public:
			DimensionMatchedFactorModel(StructurePtr s, unsigned int nParams, const std::vector<unsigned int>& pim,
				const std::vector<FactorPtr>& fs)
				: FactorModel(s, nParams, fs), paramIndexMap(pim) {}

		protected:
			FactorParametersPtr wrapParameters(const std::vector<stan::agrad::var>& params_r) const;
			std::vector<unsigned int> paramIndexMap;
		};

		class FactorTemplate
		{
		public:
			virtual void unroll(StructurePtr s, std::vector<FactorPtr>& factors) const = 0;

			// Default behavior is inefficient: unary unroll is invoked on sOld and sNew separately.
			// The convention is that all factors in fShared should refer to 'sOld.' This is enforced in
			// FactorTemplateModel::unroll, which will throw an error if this pattern is not followed.
			virtual void unroll(StructurePtr sOld, StructurePtr sNew,
				std::vector<FactorPtr>& fOld, std::vector<FactorPtr>& fNew, std::vector<FactorPtr>& fShared) const;
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
			ModelPtr unroll(StructurePtr s) const;
			void unroll(StructurePtr sOld, StructurePtr sNew, const DimensionMatchMap& dimMatch,
				ModelPtr& mOld, ModelPtr& mNew, ModelPtr& mShared) const;
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
			stan::agrad::var log_prob(const std::vector<stan::agrad::var>& params_r) const;
			std::vector<double>& getWeights() { return weights; }

		private:
			std::vector<ModelPtr> models;
			std::vector<double> weights;
		};
	}
}

#endif