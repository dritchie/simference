#ifndef __SAMPLING_H
#define __SAMPLING_H

#include <stan/mcmc/sampler.hpp>
#include <stan/mcmc/nuts.hpp>
#include <vector>

namespace simference
{
	// This is exactly like stan::mcmc::sample,
	// except it has a no-arg constructor (so it can
	// be used as a template parameter for vectors, etc.)
	class Sample
	{
	public:
		Sample(const std::vector<double>& p, double lp)
			: params(p), logprob(lp) {}
		Sample() : logprob(0.0) {}
		Sample(const stan::mcmc::sample& samp)
			: params(samp.params_r()), logprob(samp.log_prob()) {}
		std::vector<double> params;
		double logprob;
	};

	template <class Model>
	void GenerateSamples(Model& model,
						 // Initial parameters
						 const std::vector<double>& params,
						 // Store generated samples here
						 std::vector<Sample>& samples,
						 // How many iterations to run sampling for.
						 int num_iterations = 1000,
						 // How many of the above iterations count as 'warm-up' (samples discarded)
						 int num_warmup = 100,
						 // Automatically choose step size during warm-up?
						 bool epsilon_adapt = true,
						 // Keep every how many samples?
						 int num_thin = 1,
						 // Save the warm-up samples?
						 bool save_warmup = false)
	{
		stan::mcmc::nuts<boost::mt19937> sampler(model, 10, -1, 0.0, true, 0.6, 0.05, boost::mt19937(std::time(0)), &params);

		std::vector<int> params_i;	// A dummy
		sampler.set_params(params,params_i);

		if (epsilon_adapt)
		{
			sampler.adapt_on(); 
		}
		for (int m = 0; m < num_iterations; ++m)
		{
			printf("Sampling iteration %d / %d\r", m+1, num_iterations);

			Sample sample = Sample(sampler.next());

			if (m < num_warmup)
			{
				if (save_warmup && (m % num_thin) == 0)
				{
					samples.push_back(sample);
				} 
			}
			else 
			{
				if (epsilon_adapt && sampler.adapting())
				{
					sampler.adapt_off();
				}
				if (((m - num_warmup) % num_thin) != 0)
				{
					sampler.next();
				}
				else 
				{
					samples.push_back(sample);
				}
			}
		}
		printf("\n");
	}
}

#endif