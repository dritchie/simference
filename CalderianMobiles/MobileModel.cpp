#include "MobileModel.h"

using namespace std;
using namespace simference::Grammar;
using namespace simference::Math::Probability;
using namespace stan::agrad;
using namespace stan::model;
using namespace Eigen;

namespace simference
{
	namespace Models
	{
		void MobileFactorTemplate::unroll(StructurePtr s, vector<FactorPtr>& factors) const
		{
			factors.push_back(FactorPtr(new Factor(s, anchor)));
		}

		MobileFactorTemplate::Factor::Factor(StructurePtr s, const Vector3d& anchor)
			: simference::Models::Factor(s), mobile(static_pointer_cast<DerivationTree<var>>(s)->derivation, anchor)
		{
		}

		var MobileFactorTemplate::Factor::log_prob(const ParameterVector<var>& params)
		{
			auto dtree = static_pointer_cast<DerivationTree<var>>(structUnrolledFrom);
			dtree->setParams(params);
			mobile.updateAnchors();

			var lp = 0.0;

			// Static collision factors
			if (collisionsEnabled)
			{
				double rodXrodSD = 0.328407 * collisionScaleFactor;
				double rodXstringSD = 1.10272 * collisionScaleFactor;
				double rodXweightSD = 0.883831 * collisionScaleFactor;
				double weightXstringSD = 1.20902 * collisionScaleFactor;
				double weightXweightSD = 0.807079 * collisionScaleFactor;
				auto collsum = mobile.checkStaticCollisions();
				lp += NormalDistribution<var>::LogProb(collsum.rodXrod, 0.0, rodXrodSD);
				lp += NormalDistribution<var>::LogProb(collsum.rodXstring, 0.0, rodXstringSD);
				lp += NormalDistribution<var>::LogProb(collsum.rodXweight, 0.0, rodXweightSD);
				lp += NormalDistribution<var>::LogProb(collsum.weightXstring, 0.0, weightXstringSD);
				lp += NormalDistribution<var>::LogProb(collsum.weightXweight, 0.0, weightXweightSD);
			}

			// Static equilibrium factor
			if (torqueEnabled)
			{
				double torqueSD = 360.0 * torqueScaleFactor;
				lp += NormalDistribution<var>::LogProb(mobile.softMaxTorqueNorm(), 0.0, torqueSD);
			}

			return lp;
		}

		bool MobileFactorTemplate::Factor::collisionsEnabled = true;
		double MobileFactorTemplate::Factor::collisionScaleFactor = 0.33;
		bool MobileFactorTemplate::Factor::torqueEnabled = true;
		double MobileFactorTemplate::Factor::torqueScaleFactor = 0.25; // 0.001?
	}
}