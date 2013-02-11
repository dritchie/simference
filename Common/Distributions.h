#ifndef __DISTRIBUTIONS_H
#define __DISTRIBUTIONS_H

#include <stdlib.h>
#include <vector>

namespace simference
{
	template<typename T>
	class UnivariateDistribution
	{
	public:
		virtual T eval(T val) = 0;
		virtual T sample() = 0;
	};

	template<typename T>
	class UniformUnivariateDistribution
	{
	public:
		UniformUnivariateDistribution(T minv = (T)0.0, T maxv = (T)1.0)
			: minval(minv), maxval(maxv) {}
		static T Eval(T val, T minvalue, T maxvalue) { return (T)1.0 / (maxvalue - minvalue);}
		static T Sample(T minvalue = (T)0.0, T maxvalue = (T)1.0)
		{
			T t = rand() / ((T)RAND_MAX);
			return (1-t)*minvalue + t*maxvalue;
		}
		T eval (T val) { return Eval(val, minval, maxval); }
		T sample() { return Sample(minval, maxval); }

	private:
		T minval, maxval;
	};

	template<typename T>
	class MultinomialUnivariateDistribution
	{
	public:
		MultinomialUnivariateDistribution(const std::vector<T>& params)
			: parameters(params) {}
		static T Eval(unsigned int val, const std::vector<T>& params) { return params[val]; } 
		static T Sample(const std::vector<T>& params)
		{
			unsigned int result = 0;
			T x = UniformUnivariateDistribution<T>::Sample();
			T probAccum = (T) 1e-6;		// Small episilon to avoid numerical issues
			unsigned int k = params.size();
			for (; result < k; result++)
			{
				probAccum += params[result];
				if (x <= probAccum) break;
			}
			return result;
		}
		T eval(T val) { return Eval((unsigned int)val, parameters); }
		T sample() { return Sample(parameters); }
	private:
		std::vector<T> parameters;
	};
}

#endif