#include "Model.h"
#include <cassert>

using namespace stan::agrad;
using namespace std;

namespace simference
{
	vector<double> DimensionMatchMap::translateExtendedToOld(const vector<double>& extendedParams)
	{
		return translate(extendedParams, oldParamIndices);
	}

	vector<double> DimensionMatchMap::translateExtendedToNew(const vector<double>& extendedParams)
	{
		return translate(extendedParams, newParamIndices);
	}

	vector<double> DimensionMatchMap::translate(const vector<double>& extendedParams, const vector<unsigned int>& indexMap)
	{
		vector<double> result(indexMap.size());
		for (unsigned int i = 0; i < indexMap.size(); i++)
			result[i] = extendedParams[indexMap[i]];
		return result;
	}

	namespace Models
	{

		FactorModel::FactorModel(StructurePtr s, unsigned int nParams, const vector<FactorPtr>& fs)
			: Model(nParams), structUnrolledFrom(s), factors(fs)
		{
			bool allSameUnrollSource = true;
			for (auto f : factors) allSameUnrollSource &= (structUnrolledFrom == f->structUnrolledFrom);
			assert(allSameUnrollSource);
		}

		ParameterVectorPtr<var>::type FactorModel::wrapParameters(const vector<var>& params_r) const
		{
			return ParameterVectorPtr<var>::type(new ParameterVector<var>(params_r));
		}

		var FactorModel::log_prob(const vector<var>& params_r)
		{
			ParameterVectorPtr<var>::type params = wrapParameters(params_r);
			var lp = 0.0;
			for (auto f : factors)
				lp += f->log_prob(*params);
			return lp;
		}

		ParameterVectorPtr<var>::type DimensionMatchedFactorModel::wrapParameters(const vector<var>& params_r) const
		{
			return ParameterVectorPtr<var>::type(new DimensionMatchedParameterVector<var>(params_r, paramIndexMap));
		}

		void FactorTemplate::unroll(StructurePtr sOld, StructurePtr sNew,
			std::vector<FactorPtr>& fOld, std::vector<FactorPtr>& fNew, std::vector<FactorPtr>& fShared) const
		{
			// This is correct, but often inefficient.
			// Subclasses may be able to do better by looking for shared sub-structures.
			unroll(sOld, fOld);
			unroll(sNew, fNew);
		}

		void FactorTemplateModel::addTemplate(FactorTemplatePtr t)
		{
			templates.push_back(t);
		}

		void FactorTemplateModel::removeTemplate(FactorTemplatePtr t)
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

		ModelPtr FactorTemplateModel::unroll(StructurePtr s) const
		{
			vector<FactorPtr> factors;
			for (auto t : templates)
				t->unroll(s, factors);
			return ModelPtr(new FactorModel(s, s->numParams(), factors));
		}

		void FactorTemplateModel::unroll(StructurePtr sOld, StructurePtr sNew, const DimensionMatchMap& dimMatch,
			ModelPtr& mOld, ModelPtr& mNew, ModelPtr& mShared) const
		{
			unsigned int numParams = dimMatch.extendedSpaceDimension;

			vector<FactorPtr> fOld, fNew, fShared;
			for (auto t : templates)
				t->unroll(sOld, sNew, fOld, fNew, fShared);

			mOld = ModelPtr(new DimensionMatchedFactorModel(sOld, numParams, dimMatch.oldParamIndices, fOld));
			mNew = ModelPtr(new DimensionMatchedFactorModel(sNew, numParams, dimMatch.newParamIndices, fNew));
			mShared = ModelPtr(new DimensionMatchedFactorModel(sOld, numParams, dimMatch.oldParamIndices, fShared));
		}

		MixtureModel::MixtureModel(const vector<ModelPtr>& ms, const vector<double>& ws)
			: Model(ms[0]->num_params_r()), models(ms), weights(ws)
		{
			assert(ms.size() == ws.size());

			bool allModelsSameDimension = true;
			for (auto m : models) allModelsSameDimension &= (m->num_params_r() == num_params_r());
			assert(allModelsSameDimension);
		}

		MixtureModel::MixtureModel(const std::vector<ModelPtr>& ms)
			: Model(ms[0]->num_params_r()), models(ms), weights(vector<double>(ms.size(), 1.0 / ms.size()))
		{
			bool allModelsSameDimension = true;
			for (auto m : models) allModelsSameDimension &= (m->num_params_r() == num_params_r());
			assert(allModelsSameDimension);
		}

		var MixtureModel::log_prob(const vector<var>& params_r)
		{
			var lp = 0.0;
			for (unsigned int i = 0; i < models.size(); i++)
				lp += weights[i] * models[i]->log_prob(params_r);
			return lp;
		}

		void IndirectionModel::setInnerModel(ModelPtr im)
		{
			innerModel = im;
			num_params_r__ = im->num_params_r();
			param_ranges_i__.resize(im->num_params_i());
			for (unsigned int i = 0; i < im->num_params_i(); i++)
			{
				param_ranges_i__[i] = im->param_range_i(i);
			}
		}
	}
}