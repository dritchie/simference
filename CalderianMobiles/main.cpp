#include "../Common/Sampler.h"
#include "../Common/GrammarInference.h"
#include "MobileGrammar.h"
#include "Mobile.h"
#include "MobileModel.h"
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <GL/glut.h>

using namespace simference;
using namespace simference::Samplers;
using namespace simference::Models;
using namespace std;
using namespace Eigen;
using namespace stan::agrad;

//typedef double RealNum;
typedef var RealNum;

// I have to use pointers for everything because
// agrad::var cannot be statically allocated safely.
String<RealNum>::type* axiom = NULL;
shared_ptr<DerivationTree<RealNum>> derivationTree = shared_ptr<DerivationTree<RealNum>>(NULL);
Mobile<RealNum>* mobile = NULL;
Vector3d anchor(0.0, 9.5, 0.0);

vector<Sample> mostRecentSamples;
int currSampleIndex = 0;

void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-10.0, 10.0, -10.0, 10.0);
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	if (mobile)
		mobile->render();

	glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y)
{
	bool needsRedisplay = false;

	if (key == 's')
	{
		derivationTree = shared_ptr<DerivationTree<RealNum>>(new DerivationTree<RealNum>(*axiom));
		if (mobile) delete mobile;
		mobile = new Mobile<RealNum>(derivationTree->derivation, anchor);
		needsRedisplay = true;
	}
	else if (key == 'p')
	{
		FactorTemplateModel ftm;
		ftm.addTemplate(FactorTemplatePtr(new GrammarFactorTemplate));
		ftm.addTemplate(FactorTemplatePtr(new MobileFactorTemplate(anchor)));
		ModelPtr model = ftm.unroll(derivationTree);
		vector<var> p; derivationTree->getParams(p);
		vector<double> params; for (auto d : p) params.push_back(d.val());
		cout << "log prob: " << model->log_prob(params) << endl;
	}
	else if (key == 'c')
	{
		if (mobile)
		{
			auto record = mobile->checkStaticCollisions();
			record.print();
		}
	}
	else if (key == 'd')
	{
		if (mobile)
			cout << "Ancestor/descendant sanity check passed: " << mobile->sanityCheckNodeCodes() << endl;
	}
	else if (key == 'n')
	{
		if (mobile)
			mobile->printNodeCodes();
	}
	else if (key == 't')
	{
		if (mobile)
			cout << "Softmax rod torque norm: " << mobile->softMaxTorqueNorm() << endl;
			//cout << "Average rod torque norm: " << mobile->netTorqueNorm() << endl;
	}
	else if (key == 'a')
	{
		// Sample a bunch of derivations and average the collision stats
		// as well as the net torque norm
		// (Useful for deciding factor kernel bandwidths)
		Mobile<RealNum>::CollisionSummary summ;
		RealNum torque = 0.0;
		static const unsigned int nCollisionSamples = 1000;
		for (unsigned int i = 0; i < nCollisionSamples; i++)
		{
			DerivationTree<RealNum> dtree(*axiom);
			auto& dstring = dtree.derivation;
			auto dmobile = new Mobile<RealNum>(dstring, anchor);
			auto collsum = dmobile->checkStaticCollisions();
			torque += dmobile->softMaxTorqueNorm();
			//torque += dmobile->netTorqueNorm();
			delete dmobile;
			summ.rodXrod += collsum.rodXrod;
			summ.rodXstring += collsum.rodXstring;
			summ.rodXweight += collsum.rodXweight;
			summ.weightXstring += collsum.weightXstring;
			summ.weightXweight += collsum.weightXweight;
		}
		summ.rodXrod /= nCollisionSamples;
		summ.rodXstring /= nCollisionSamples;
		summ.rodXweight /= nCollisionSamples;
		summ.weightXstring /= nCollisionSamples;
		summ.weightXweight /= nCollisionSamples;
		summ.print();
		cout << "torque: " << torque / nCollisionSamples << endl;
	}
	else if (key == 'h')
	{
		// Use stan's hmc to sample a bunch of parameter settings
		// for the current derived structure.

		static const unsigned int numHmcIters = 1000;
		static const unsigned int numWarmup = 100;

		vector<var> params;
		derivationTree->getParams(params);
		vector<double> initParams;
		for (auto var : params) initParams.push_back(var.val());

		FactorTemplateModel ftm;
		ftm.addTemplate(FactorTemplatePtr(new GrammarFactorTemplate));
		ftm.addTemplate(FactorTemplatePtr(new MobileFactorTemplate(anchor)));
		ModelPtr model = ftm.unroll(derivationTree);
		vector<Sample> samples;
		DiffusionSampler sampler(StructurePtr(NULL), *model, initParams);
		Sampler::sample(sampler, samples, numHmcIters, numWarmup);

		sampler.writeAnalytics(cout);

		// Find the sample with highest log-probability and display that state
		sort(samples.begin(), samples.end(), [](const Sample& s1, const Sample& s2) { return s1.logprob > s2.logprob; });
		//unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		//shuffle(samples.begin(), samples.end(), default_random_engine(seed));
		const Sample& bestsamp = samples[0];
		params.clear();
		for (double d : bestsamp.params) params.push_back(var(d));
		derivationTree->setParams(params);
		mobile->updateAnchors();
		needsRedisplay = true;
	}
	else if (key == 'j')
	{
		// Test out jump proposals
		vector<var> params; derivationTree->getParams(params);
		vector<double> p; for (auto var : params) p.push_back(var.val());
		FactorTemplateModelPtr ftmp = FactorTemplateModelPtr(new FactorTemplateModel);
		ftmp->addTemplate(FactorTemplatePtr(new GrammarFactorTemplate));
		GrammarJumpSampler gs(ftmp, derivationTree, p);

		// We have to reset the original params because when we initialized the sampler, it
		// computed some gradients, which clobbers the vars in the derivationTree.
		params.clear(); for (auto d : p) params.push_back(d);
		derivationTree->setParams(params);

		// Make proposal
		DimensionMatchMap matching;
		vector<double> p2;
		StructurePtr newstruct = gs.jumpProposalTest(p2, matching);

		// Visualize the result
		derivationTree = static_pointer_cast<DerivationTree<RealNum>>(newstruct);
		if (mobile) delete mobile;
		mobile = new Mobile<RealNum>(derivationTree->derivation, anchor);
		needsRedisplay = true;
	}
	else if (key == 'l')
	{
		static const unsigned int numLARJiters = 400/*1000*/;	// RESET this to 1000
		static const unsigned int numLARJannealSteps = 40;
		static const double jumpFreq = 0.05/*1.0*/;				// RESET this to 0.1

		// Test LARJ sampling
		vector<var> params; derivationTree->getParams(params);
		vector<double> p; for (auto var : params) p.push_back(var.val());
		FactorTemplateModelPtr ftmp = FactorTemplateModelPtr(new FactorTemplateModel);
		ftmp->addTemplate(FactorTemplatePtr(new GrammarFactorTemplate));
		ftmp->addTemplate(FactorTemplatePtr(new MobileFactorTemplate(anchor)));
		GrammarJumpSampler gs(ftmp, derivationTree, p, numLARJannealSteps, jumpFreq);

		mostRecentSamples.clear();
		JumpSampler::sample(gs, mostRecentSamples, numLARJiters);

		gs.writeAnalytics(cout);
	}
	else if (key == 'v')
	{
		// 'cross-validate' a bunch of different LARJ parameter choices
		// - warm up adaptation VS continuous adaptation
		// - broader distributions VS narrower distributions
		// - more annealing steps VS fewer annealing steps
		// - collisions ON/OFF
		// - torque ON/OFF

		FactorTemplateModelPtr ftmp = FactorTemplateModelPtr(new FactorTemplateModel);
		ftmp->addTemplate(FactorTemplatePtr(new GrammarFactorTemplate));
		ftmp->addTemplate(FactorTemplatePtr(new MobileFactorTemplate(anchor)));
		vector<var> params; derivationTree->getParams(params);
		vector<double> p; for (auto var : params) p.push_back(var.val());

		string adaptTypes[2] = {"WarmUp", "Continuous"};
		double scaleMultipliers[5] = { 0.1, 0.5, 1.0, 2.0, 10.0 };
		unsigned int numAnnealingSteps[3] = { 20, 50, 100 };
		//bool collisionsEnabled[2] = { true, false };
		//bool torqueEnabled[2] = { true, false };
		bool collisionsEnabled[1] = { true };
		bool torqueEnabled[1] = { true };

		// Invariant parameters
		double jumpFrequency = 0.05;
		unsigned int numIterations = 1000;
		unsigned int numWarmup = 100;

		double originalCollisionScale = MobileFactorTemplate::Factor::collisionScaleFactor;
		double originalTorqueScale = MobileFactorTemplate::Factor::torqueScaleFactor;

		// Initialize log
		ofstream log("log.csv");
		log << "Adapt,Scale,AnnealSteps,Collisions,Torque,DiffusionAccept,AnnealAccept,JumpAccept" << endl;
		log.close();

		// Initialize sample log
		ofstream samples("samples.csv");
		samples << "Adapt,Scale,AnnealSteps,Collisions,Torque,SampleType,Accepted,IterationNumber" << endl;

		for (auto adaptType : adaptTypes)
		{
			for (auto scaleMult : scaleMultipliers)
			{
				for (auto nAnneal : numAnnealingSteps)
				{
					for (auto collOn : collisionsEnabled)
					{
						for (auto torqueOn : torqueEnabled)
						{
							// Enable/disable factor components
							MobileFactorTemplate::Factor::collisionsEnabled = collOn;
							MobileFactorTemplate::Factor::torqueEnabled = torqueOn;

							// Set scales
							MobileFactorTemplate::Factor::collisionScaleFactor = originalCollisionScale*scaleMult;
							MobileFactorTemplate::Factor::torqueScaleFactor = originalTorqueScale*scaleMult;

							// Set up and run sampler
							GrammarJumpSampler gs(ftmp, derivationTree, p, nAnneal, jumpFrequency);
							vector<Sample> samples;
							if (adaptType == "WarmUp")
								JumpSampler::sampleWithWarmup(gs, samples, numIterations + numWarmup, numWarmup);
							else if (adaptType == "Continuous")
								JumpSampler::sample(gs, samples, numIterations);

							// Report
							cout << "Adapt=" << adaptType << ", Scale=" << scaleMult << ", AnnealSteps=" << nAnneal << ", Collisions=" << collOn << ", Torque=" << torqueOn << endl;
							gs.writeAnalytics(cout);
							ofstream log("log.csv", std::ios_base::app);
							log << adaptType << "," << scaleMult << "," << nAnneal << "," << collOn << "," << torqueOn << ","
								<< gs.diffusionAcceptanceRatio() << "," << gs.annealingAcceptanceRatio() << "," << gs.jumpAcceptanceRatio() << endl;
							log.close();

							// Spit out all samples
							ofstream samps("samples.csv", ios_base::app);
							for (unsigned int i = 0; i < samples.size(); i++)
							{
								auto& s = samples[i];
								if (s.proposalType == Sample::Diffusion || s.proposalType == Sample::Annealing || s.proposalType == Sample::JumpEnd)
								{
									string stype = (s.proposalType == Sample::Diffusion ? "Diffusion" : (s.proposalType == Sample::Annealing ? "Annealing" : "Jump"));
									samps << adaptType << "," << scaleMult << "," << nAnneal << "," << collOn << "," << torqueOn << ","
										<< stype << "," << s.accepted << "," << i << endl;
								}
							}
							samps.close();
						}
					}
				}
			}
		}
	}

	if (needsRedisplay)
		glutPostRedisplay();
}

void special(int key, int x, int y)
{
	bool needsRedisplay = false;

	auto displayCurrentSample = [&needsRedisplay] ()
	{
		const Sample& samp = mostRecentSamples[currSampleIndex];
		derivationTree = static_pointer_cast<DerivationTree<var>>(samp.structure);
		vector<var> params;
		for (double d : samp.params) params.push_back(d);
		derivationTree->setParams(params);
		delete mobile;
		mobile = new Mobile<var>(derivationTree->derivation, anchor);
		needsRedisplay = true;
		cout << "(" << currSampleIndex << ") ";
		samp.print(cout);
	};

	if (key == GLUT_KEY_LEFT)
	{
		if (mostRecentSamples.size() > 0)
		{
			currSampleIndex = (currSampleIndex - 1);
			if (currSampleIndex < 0) currSampleIndex = mostRecentSamples.size() - 1;
			displayCurrentSample();
		}
	}
	else if (key == GLUT_KEY_RIGHT)
	{
		currSampleIndex = (currSampleIndex + 1) % mostRecentSamples.size();
		displayCurrentSample();
	}

	if (needsRedisplay)
		glutPostRedisplay();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitWindowSize(800, 800);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutCreateWindow("Calderian Mobiles");
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);

	axiom = new String<RealNum>::type;

	// We start with a single string (the string from which everything hangs)
	auto root = new StringTerminal<RealNum>(0, 0);
	root->params[StringLength] = 2.0;
	axiom->push_back(SymbolPtr<RealNum>::type(root));
	axiom->push_back(SymbolPtr<RealNum>::type(new StringEndpointVariable<RealNum>(0)));

	glutMainLoop();

	return 0;
}