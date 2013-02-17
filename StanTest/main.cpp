#include <stan/model/prob_grad_ad.hpp>
#include <stan/gm/command.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

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

	// Sample from the model
	gm::sample_from(sampler, true, 0, 2000, 100, 0, false, cout, params_r, params_i, m);

	return 0;
}