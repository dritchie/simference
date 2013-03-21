#ifndef __MOBILE_MODEL_H
#define __MOBILE_MODEL_H

#include "../Common/Model.h"
#include "Mobile.h"

namespace simference
{
	namespace Models
	{
		class MobileFactorTemplate : public FactorTemplate
		{
		public:
			MobileFactorTemplate(const Eigen::Vector3d& a) : anchor(a) {}
			void unroll(StructurePtr s, std::vector<FactorPtr>& factors) const;

			class Factor : public simference::Models::Factor
			{
			public:
				Factor(StructurePtr s, const Eigen::Vector3d& anchor);
				stan::agrad::var log_prob(const ParameterVector<stan::agrad::var>& params);

				static bool collisionsEnabled;
				static double collisionScaleFactor;
				static bool torqueEnabled;
				static double torqueScaleFactor;

			private:
				Mobile<stan::agrad::var> mobile;
			};
		private:
			Eigen::Vector3d anchor;
		};
	}
}

#endif