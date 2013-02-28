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
		MobileModel(Grammar::DerivationTree<stan::agrad::var>& dtree,
			        const Eigen::Vector3d& anchor)
			:
		stan::model::prob_grad_ad(dtree.numParams()),
		derivationTree(dtree),
		mobile(dtree.derivation, anchor) {}

		stan::agrad::var log_prob(
					std::vector<stan::agrad::var>& params_r, 
					std::vector<int>& params_i,
					std::ostream* output_stream = 0);
	private:
		Grammar::DerivationTree<stan::agrad::var>& derivationTree;
		Mobile<stan::agrad::var> mobile;
	};
}

#endif