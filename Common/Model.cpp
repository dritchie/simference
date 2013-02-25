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
				lp += f->log_prob(params_r.begin() + f->parameterOffset());
			return lp;
		}

		MixtureModel::MixtureModel(const vector<ModelPtr>& ms, const vector<double>& ws)
			: Model(ms[0]->num_params_r()), models(ms), weights(ws)
		{
			assert(ms.size() == ws.size());

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