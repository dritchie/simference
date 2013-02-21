#ifndef __DISTRIBUTIONS_H
#define __DISTRIBUTIONS_H

#include "Math.h"
#include <stdlib.h>
#include <vector>
#include <limits>

namespace simference
{
	namespace Math
	{
		namespace Probability
		{
			template<typename ProbType, typename ValType = ProbType>
			class Distribution
			{
			public:
				virtual ProbType prob(ValType val) const = 0;
				ProbType logprob(ValType val) const { return (ProbType)log(prob(val)); }
				virtual ValType sample() const = 0;
			};

			template<typename ValProbType, typename ParamType = ValProbType>
			class UniformDistribution : public Distribution<ValProbType, ValProbType>
			{
			public:
				UniformDistribution(ParamType minv = (ParamType)0.0, ParamType maxv = (ParamType)1.0)
					: minval(minv), maxval(maxv) {}
				static ValProbType Prob(ValProbType val, ParamType minvalue, ParamType maxvalue)  { return (ValProbType)1.0 / (maxvalue - minvalue);}
				static ValProbType Sample(ParamType minvalue = (ParamType)0.0, ParamType maxvalue = (ParamType)1.0)
				{
					ValProbType t = rand() / ((ValProbType)RAND_MAX);
					return (1-t)*minvalue + t*maxvalue;
				}
				ValProbType prob (ValProbType val) const { return Prob(val, minval, maxval); }
				ValProbType sample() const { return Sample(minval, maxval); }

			private:
				ParamType minval, maxval;
			};

			template<typename ProbType, typename ParamType = ProbType>
			class MultinomialDistribution : public Distribution<ProbType, unsigned int>
			{
			public:
				MultinomialDistribution(const std::vector<ParamType>& params)
					: parameters(params) {}
				static ProbType Prob(unsigned int val, const std::vector<ParamType>& params) { return (ProbType)params[val]; } 
				static unsigned int Sample(const std::vector<ParamType>& params)
				{
					unsigned int result = 0;
					ProbType x = UniformDistribution<ProbType>::Sample();
					ProbType probAccum = (ProbType) 1e-6;		// Small episilon to avoid numerical issues
					unsigned int k = params.size();
					for (; result < k; result++)
					{
						probAccum += params[result];
						if (x <= probAccum) break;
					}
					return result;
				}
				ProbType prob(unsigned int val) const { return Prob(val, parameters); }
				unsigned int sample() const { return Sample(parameters); }
			private:
				std::vector<ParamType> parameters;
			};

			template<typename ValProbType, typename ParamType = ValProbType>
			class NormalDistribution : public Distribution<ValProbType, ValProbType>
			{
			public:
				NormalDistribution(ParamType mu = (ParamType)0.0, ParamType sigma = (ParamType)1.0)
					: mean(mu), stddev(sigma) {}
				static ValProbType Prob(ValProbType val, ParamType mu, ParamType sigma)
				{
					ParamType valMinusMu = val - mu;
					return (ValProbType)1.0/(sigma*sqrt(TwoPi)) * exp(-valMinusMu*valMinusMu/(2*sigma*sigma));
				}
				static ValProbType Sample(ParamType mu = (ParamType)0.0, ParamType sigma = (ParamType)1.0)
				{
					// Box-Muller method
					ValProbType minval = (ValProbType)(0.0) + std::numeric_limits<ValProbType>::min();
					ValProbType maxval = (ValProbType)(1.0) - std::numeric_limits<ValProbType>::min();
					ValProbType u = UniformDistribution<ValProbType>::Sample(minval, maxval);
					ValProbType v = UniformDistribution<ValProbType>::Sample(minval, maxval);
					ValProbType result = (ValProbType)(sqrt(-2 * log(u)) * cos(2*Pi*v));
					// Note: sqrt(-2 * log(u)) * sin(2*PI*v) is also a valid choice
					return mu + sigma*result;

				}
				ValProbType prob(ValProbType val) const { return Prob(val, mean, stddev); }
				ValProbType sample() const { return Sample(mean, stddev); }
			private:
				ParamType mean, stddev;
			};

			static double erf_a = 0.147;
			
			template<typename T>
			T erf(T x)
			{
				T x2 = x*x;
				T inner = exp(-1*x2*(4/Pi + erf_a*x2) / (1 + erf_a*x2));
				return signum(x)*sqrt(1 - inner);
			}

			template<typename T>
			T erfi(T x)
			{
				T x2 = x*x;
				T i1 = 2/(Pi*erf_a) + log(1-x2)/2;
				T i2 = log(1-x2)/erf_a;
				return signum(x)*sqrt(sqrt(i1*i1 - i2) - i1);
			}

			template<typename T>
			T cumulativeNormal(T x)
			{
				return 0.5 * (1 + erf(x / sqrt(2.0)));
			}

			template<typename T>
			T invCumulativeNormal(T x)
			{
				return sqrt(2.0) * erfi(2*x - 1);
			}

			template<typename ValProbType, typename ParamType = ValProbType>
			class TruncatedNormalDistribution : public Distribution<ValProbType, ValProbType>
			{
			public:
				TruncatedNormalDistribution(ParamType mu, ParamType sigma, ParamType lo, ParamType hi)
					: mean(mu), stddev(sigma), lowerBound(lo), upperBound(hi) {}
				static ValProbType Prob(ValProbType val, ParamType mu, ParamType sigma, ParamType lo, ParamType hi)
				{
					if (val > lo && val < hi)
					{
						ValProbType numer = NormalDistribution<ValProbType>::Prob(val, mu, sigma);
						ValProbType denom = cumulativeNormal((hi-mu)/sigma) - cumulativeNormal((lo-mu)/sigma);
						return numer / denom;
					}
					else return 0.0;
				}
				static ValProbType Sample(ParamType mu, ParamType sigma, ParamType lo, ParamType hi)
				{
					ValProbType cumNormLo = cumulativeNormal(lo);
					ValProbType raw = invCumulativeNormal(cumNormLo + UniformDistribution<ValProbType>::Sample() * (cumulativeNormal(hi) - cumNormLo));
					return sigma*raw + mu;
				}
				ValProbType prob(ValProbType val) const { return Prob(val, mean, stddev, lowerBound, upperBound); }
				ValProbType sample() const { return Sample(mean, stddev, lowerBound, upperBound); }
			private:
				ParamType mean, stddev, lowerBound, upperBound;
			};
		}
	}
}

#endif