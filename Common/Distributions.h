#ifndef __DISTRIBUTIONS_H
#define __DISTRIBUTIONS_H

#include <stdlib.h>
#include <vector>
#include <limits>

namespace simference
{
	#define PI 3.141592654
	#define TWO_PI 6.283185308

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
			return (T)1.0/(sigma*sqrt(TWO_PI)) * exp(-valMinusMu*valMinusMu/(2*sigma*sigma));
		}
		static T Sample(T mu = (T)0.0, T sigma = (T)1.0)
		{
			// Box-Muller method
			T minval = (T)(0.0) + std::numeric_limits<T>::min();
			T maxval = (T)(1.0) - std::numeric_limits<T>::min();
			T u = UniformDistribution<T>::Sample(minval, maxval);
			T v = UniformDistribution<T>::Sample(minval, maxval);
			T result = (T)(sqrt(-2 * log(u)) * cos(2*PI*v));
			// Note: sqrt(-2 * log(u)) * sin(2*PI*v) is also a valid choice
			return mu + sigma*result;

		}
		T prob(T val) { return Prob(val, mean, stddev); }
		T sample() { return Sample(mean, stddev); }
	private:
		T mean, stddev;
	};
}

#endif