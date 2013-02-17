#include <stan/model/prob_grad_ad.hpp>
#include <stan/gm/command.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

#include <fstream>

using namespace std;
using namespace stan;

class TestModel : public model::prob_grad_ad
{
public:
	TestModel(double mu, double sigma)
		: model::prob_grad_ad(1), mean(mu), stddev(sigma) {}
	agrad::var log_prob(
		std::vector<agrad::var>& params_r, 
		std::vector<int>& params_i,
		std::ostream* output_stream = 0)
	{
		agrad::var lp__(0.0);
		lp__ += prob::normal_log(params_r[0], mean, stddev);
		return lp__;
	}
private:
	double mean, stddev;
};

int main(int argc, char** argv)
{
	// Create a model
	TestModel m(0.0, 1.0);

	// Create a sampler
	auto sampler = mcmc::nuts<>(m);

	// Initialize parameter vectors
	vector<double> params_r(1, 0.0);
	vector<int> params_i;

	// Open up a CSV file for output writing.
	ofstream outfile("samples.csv");

	// Sample from the model
	gm::sample_from(sampler,	// The sampler to use
					true,		// Whether to use adaptation to find the best step size epsilon
					0,			// How many iterations to take in between printing updates to the console
					2000,		// Number of iterations to run the sampler for
					100,		// 'num_warmup' , which I assume means the number of burn-in iterations
					1,			// 'num_thin' -- has to do with how many itermediate samples we discard between kept ones
					false,		// 'save_warmup' -- whether the samples generated during burn-in should be saved
					outfile,	// The file stream where sample output should be written
					params_r,	// Real-valued parameters for the model -- these are the inital values
					params_i,	// Integer-valued parameters -- these are the initial values
					m);			// The model being sampled from

	outfile.close();
	return 0;
}