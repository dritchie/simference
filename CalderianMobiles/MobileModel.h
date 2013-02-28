#ifndef __MOBILE_MODEL_H
#define __MOBILE_MODEL_H

#include "../Common/Model.h"
#include "Mobile.h"

namespace simference
{
	// For a fixed derivation from the grammar, this model
	// returns the log probability of parameter settings to that derivation.
	class MobileModel : public Models::Model
	{
	public:
		MobileModel(Grammar::DerivationTree<stan::agrad::var>& dtree,
			        const Eigen::Vector3d& anchor)
			:
		Model(dtree.numParams()),
		derivationTree(dtree),
		mobile(dtree.derivation, anchor) {}

		stan::agrad::var log_prob(const std::vector<stan::agrad::var>& params_r);
	private:
		Grammar::DerivationTree<stan::agrad::var>& derivationTree;
		Mobile<stan::agrad::var> mobile;
	};
}

#endif