#include "Model.h"
#include <cassert>

using namespace stan::agrad;
using namespace std;

namespace simference
{
	namespace Models
	{
		void FactorModel::addFactor(FactorPtr f)
		{
			factors.push_back(f);
		}

		void FactorModel::removeFactor(FactorPtr f)
		{
			for (auto it = factors.begin(); it != factors.end(); it++)
			{
				if (*it == f)
				{
					factors.erase(it);
					return;
				}
			}
		}

		var FactorModel::log_prob(vector<var>& params_r)
		{
			var lp = 0.0;
			for (auto f : factors)
				lp += f->log_prob(params_r);
			return lp;
		}

		void FactorTemplate::unroll(StructurePtr sOld, StructurePtr sNew,
				std::vector<FactorPtr>& fOld, std::vector<FactorPtr>& fNew, std::vector<FactorPtr>& fShared)
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
		void FactorTemplateModel::unroll(StructurePtr s, std::vector<FactorPtr>& factors)
		{
			for (auto t : templates)
				t->unroll(s, factors);
		}

		void FactorTemplateModel::unroll(StructurePtr sOld, StructurePtr sNew,
				std::vector<FactorPtr>& fOld, std::vector<FactorPtr>& fNew, std::vector<FactorPtr>& fShared)
		{
			for (auto t : templates)
				t->unroll(sOld, sNew, fOld, fNew, fShared);
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

		var MixtureModel::log_prob(vector<var>& params_r)
		{
			var lp = 0.0;
			for (unsigned int i = 0; i < models.size(); i++)
				lp += weights[i] * models[i]->log_prob(params_r);
			return lp;
		}
	}
}