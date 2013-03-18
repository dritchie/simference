#include "../Common/Sampler.h"
#include "../Common/GrammarInference.h"
#include "MobileGrammar.h"
#include "Mobile.h"
#include "MobileModel.h"
#include <iostream>
#include <chrono>
#include <random>
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

		//cout << "param log prob: " << derivationTree->paramLogProb() << endl;

		needsRedisplay = true;
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
		sampler.sample(samples, numHmcIters, numWarmup);

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

		cout << "param log prob: " << derivationTree->paramLogProb() << endl;
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
		static const unsigned int numLARJiters = 5;	// RESET this to 1000
		static const unsigned int numLARJwarmup = 0;	// RESET this to 100
		static const unsigned int numLARJannealSteps = 20;
		static const double jumpFreq = 1.0;				// RESET this to 0.1

		// Test LARJ sampling
		vector<var> params; derivationTree->getParams(params);
		vector<double> p; for (auto var : params) p.push_back(var.val());
		FactorTemplateModelPtr ftmp = FactorTemplateModelPtr(new FactorTemplateModel);
		ftmp->addTemplate(FactorTemplatePtr(new GrammarFactorTemplate));
		//ftmp->addTemplate(FactorTemplatePtr(new MobileFactorTemplate(anchor)));
		GrammarJumpSampler gs(ftmp, derivationTree, p, numLARJannealSteps, jumpFreq);

		mostRecentSamples.clear();
		gs.sample(mostRecentSamples, numLARJiters, numLARJwarmup);

		gs.writeAnalytics(cout);
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