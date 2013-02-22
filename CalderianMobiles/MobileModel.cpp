#include "MobileModel.h"

using namespace std;
using namespace simference::Grammar;
using namespace simference::Math::Probability;
using namespace stan::agrad;
using namespace stan::model;

namespace simference
{
	var MobileModel::log_prob(
					vector<var>& params_r, 
					vector<int>& params_i,
					ostream* output_stream)
	{
		var lp = 0.0;

		derivation.setParams(params_r);
		mobile.updateAnchors();

		lp += derivation.paramLogProb();

		// Static collision factors
		static const double collisionScaleFactor = 0.01;
		static const double rodXrodSD = 0.328407 * collisionScaleFactor;
		static const double rodXstringSD = 1.10272 * collisionScaleFactor;
		static const double rodXweightSD = 0.883831 * collisionScaleFactor;
		static const double weightXstringSD = 1.20902 * collisionScaleFactor;
		static const double weightXweightSD = 0.807079 * collisionScaleFactor;
		auto collsum = mobile.checkStaticCollisions();
		lp += NormalDistribution<var>::LogProb(collsum.rodXrod, 0.0, rodXrodSD);
		lp += NormalDistribution<var>::LogProb(collsum.rodXstring, 0.0, rodXstringSD);
		lp += NormalDistribution<var>::LogProb(collsum.rodXweight, 0.0, rodXweightSD);
		lp += NormalDistribution<var>::LogProb(collsum.weightXstring, 0.0, weightXstringSD);
		lp += NormalDistribution<var>::LogProb(collsum.weightXweight, 0.0, weightXweightSD);

		// Static equilibrium factor
		//static const double eqScaleFactor = 0.15;
		static const double eqScaleFactor = 0.001;
		//static const double torqueSD = 135.0 * eqScaleFactor;
		static const double torqueSD = 360.0 * eqScaleFactor;
		//lp += NormalDistribution<var>::LogProb(mobile.netTorqueNorm(), 0.0, torqueSD);
		lp += NormalDistribution<var>::LogProb(mobile.softMaxTorqueNorm(), 0.0, torqueSD);

		return lp;
	}
}