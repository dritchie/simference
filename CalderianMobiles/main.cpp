#include <stan/agrad/agrad.hpp>
#include <stan/mcmc/nuts.hpp>

#include "MobileGrammar.h"
#include "Mobile.h"
#include "MobileModel.h"
#include "../Common/Sampling.h"
#include <iostream>
#include <GL/glut.h>

using namespace simference;
using namespace std;
using namespace Eigen;
using namespace stan::agrad;

//typedef double RealNum;
typedef var RealNum;

// I have to use pointers for everything because
// agrad::var cannot be statically allocated safely.
String<RealNum>* axiom = NULL;
String<RealNum>* derivedString = NULL;
Mobile<RealNum>* mobile = NULL;
Mobile<RealNum>::Vector3r* anchor = NULL;

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
		auto dtree = Derive(*axiom);
		*derivedString = dtree.derivedString();
		if (mobile) delete mobile;
		mobile = new Mobile<RealNum>(*derivedString, *anchor);

		//auto* params = Parameters<RealNum>::Instance();

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
			cout << "Sum of rod torque norms: " << mobile->netTorqueNorm() << endl;
	}
	else if (key == 'a')
	{
		// Sample a bunch of derivations and average the collision stats
		// Report 1/3 the average
		// (Useful for deciding factor kernel bandwidths)
		Mobile<RealNum>::CollisionSummary summ;
		static const unsigned int nCollisionSamples = 300;
		for (unsigned int i = 0; i < nCollisionSamples; i++)
		{
			auto dtree = Derive(*axiom);
			auto dstring = dtree.derivedString();
			auto dmobile = new Mobile<RealNum>(dstring, *anchor);
			auto collsum = dmobile->checkStaticCollisions();
			delete dmobile;
			summ.rodXrod += collsum.rodXrod;
			summ.rodXstring += collsum.rodXstring;
			summ.rodXweight += collsum.rodXweight;
			summ.weightXstring += collsum.weightXstring;
			summ.weightXweight += collsum.weightXweight;
		}
		summ.rodXrod *= 0.333 / nCollisionSamples;
		summ.rodXstring *= 0.333 / nCollisionSamples;
		summ.rodXweight *= 0.333 / nCollisionSamples;
		summ.weightXstring *= 0.333 / nCollisionSamples;
		summ.weightXweight *= 0.333 / nCollisionSamples;
		summ.print();
	}
	else if (key == 'h')
	{
		// Use stan's hmc to sample a bunch of parameter settings
		// for the current derived structure.

		static const unsigned int numHmcIters = 100;
		static const unsigned int numWarmup = 10;

		vector<var> params;
		derivedString->getParams(params);
		vector<double> initParams;
		for (auto var : params) initParams.push_back(var.val());

		auto model = MobileModel(*derivedString, *anchor);
		auto sampler = stan::mcmc::nuts<>(model);
		vector<Sample> samples;
		GenerateSamples(sampler, initParams, samples, numHmcIters, numWarmup);

		// Find the sample with highest log-probability and display that state
		std::sort(samples.begin(), samples.end(), [](const Sample& s1, const Sample& s2) { return s1.logprob > s2.logprob; });
		const Sample& bestsamp = samples[0];
		params.clear();
		for (double d : bestsamp.params) params.push_back(var(d));
		derivedString->setParams(params);
		mobile->updateAnchors();
		needsRedisplay = true;
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

	axiom = new String<RealNum>;
	derivedString = new String<RealNum>;
	anchor = new Mobile<RealNum>::Vector3r(0.0, 9.5, 0.0);

	// We start with a single string (the string from which everything hangs)
	auto root = new StringTerminal<RealNum>(0);
	root->params[0] = 2.0;
	axiom->symbols.push_back(SymbolPtr(root));
	axiom->symbols.push_back(SymbolPtr(new StringEndpointVariable<RealNum>(0)));

	glutMainLoop();

	return 0;
}