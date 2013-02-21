#ifndef __MOBILE_MODEL_H
#define __MOBILE_MODEL_H

#include <stan/model/prob_grad_ad.hpp>
#include "Mobile.h"

namespace simference
{
	// For a fixed derivation from the grammar, this model
	// returns the log probability of parameter settings to that derivation.
	class MobileModel : public stan::model::prob_grad_ad
	{
	public:
		MobileModel(const Grammar::String<stan::agrad::var>& str,
			        const Eigen::Vector3d& anchor)
			:
		stan::model::prob_grad_ad(str.numParams()),
		derivation(str),
		mobile(str, anchor) {}

		stan::agrad::var log_prob(
					std::vector<stan::agrad::var>& params_r, 
					std::vector<int>& params_i,
					std::ostream* output_stream = 0);
	private:
		Grammar::String<stan::agrad::var> derivation;
		Mobile<stan::agrad::var> mobile;
	};
}

#endif