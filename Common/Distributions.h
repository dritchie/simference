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
			template<typename T>
			class Distribution
			{
			public:
				virtual T prob(T val) = 0;
				T logprob(T val) { return (T)log(prob(val)); }
				virtual T sample() = 0;
			};

			template<typename T>
			class UniformDistribution : public Distribution<T>
			{
			public:
				UniformDistribution(T minv = (T)0.0, T maxv = (T)1.0)
					: minval(minv), maxval(maxv) {}
				static T Prob(T val, T minvalue, T maxvalue) { return (T)1.0 / (maxvalue - minvalue);}
				static T Sample(T minvalue = (T)0.0, T maxvalue = (T)1.0)
				{
					T t = rand() / ((T)RAND_MAX);
					return (1-t)*minvalue + t*maxvalue;
				}
				T prob (T val) { return Prob(val, minval, maxval); }
				T sample() { return Sample(minval, maxval); }

			private:
				T minval, maxval;
			};

			template<typename T>
			class MultinomialDistribution : public Distribution<T>
			{
			public:
				MultinomialDistribution(const std::vector<T>& params)
					: parameters(params) {}
				static T Prob(unsigned int val, const std::vector<T>& params) { return params[val]; } 
				static T Sample(const std::vector<T>& params)
				{
					unsigned int result = 0;
					T x = UniformDistribution<T>::Sample();
					T probAccum = (T) 1e-6;		// Small episilon to avoid numerical issues
					unsigned int k = params.size();
					for (; result < k; result++)
					{
						probAccum += params[result];
						if (x <= probAccum) break;
					}
					return result;
				}
				T prob(T val) { return Prob((unsigned int)val, parameters); }
				T sample() { return Sample(parameters); }
			private:
				std::vector<T> parameters;
			};

			template<typename T>
			class NormalDistribution : public Distribution<T>
			{
			public:
				NormalDistribution(T mu = (T)0.0, T sigma = (T)1.0)
					: mean(mu), stddev(sigma) {}
				static T Prob(T val, T mu, T sigma)
				{
					T valMinusMu = val - mu;
					return (T)1.0/(sigma*sqrt(TwoPi)) * exp(-valMinusMu*valMinusMu/(2*sigma*sigma));
				}
				static T Sample(T mu = (T)0.0, T sigma = (T)1.0)
				{
					// Box-Muller method
					T minval = (T)(0.0) + std::numeric_limits<T>::min();
					T maxval = (T)(1.0) - std::numeric_limits<T>::min();
					T u = UniformDistribution<T>::Sample(minval, maxval);
					T v = UniformDistribution<T>::Sample(minval, maxval);
					T result = (T)(sqrt(-2 * log(u)) * cos(2*Pi*v));
					// Note: sqrt(-2 * log(u)) * sin(2*PI*v) is also a valid choice
					return mu + sigma*result;

				}
				T prob(T val) { return Prob(val, mean, stddev); }
				T sample() { return Sample(mean, stddev); }
			private:
				T mean, stddev;
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

			template<typename T>
			class TruncatedNormalDistribution : public Distribution<T>
			{
			public:
				TruncatedNormalDistribution(T mu, T sigma, T lo, T hi)
					: mean(mu), stddev(sigma), lowerBound(lo), upperBound(hi) {}
				static T Prob(T val, T mu, T sigma, T lo, T hi)
				{
					if (val >= lo && val <= hi)
					{
						T numer = NormalDistribution<T>::Prob(val, mu, sigma);
						T denom = cumulativeNormal((hi-mu)/sigma) - cumulativeNormal((lo-mu)/sigma);
						return numer / denom;
					}
					else return 0.0;
				}
				static T Sample(T mu, T sigma, T lo, T hi)
				{
					T cumNormLo = cumulativeNormal(lo);
					T raw = invCumulativeNormal(cumNormLo + UniformDistribution<T>::Sample() * (cumulativeNormal(hi) - cumNormLo));
					return sigma*raw + mu;
				}
				T prob(T val) { return Prob(val, mean, stddev, lowerBound, upperBound); }
				T sample() { return Sample(mean, stddev, lowerBound, upperBound); }
			private:
				T mean, stddev, lowerBound, upperBound;
			};
		}
	}
}

#endif