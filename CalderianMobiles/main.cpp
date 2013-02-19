#include "MobileGrammar.h"
#include "Mobile.h"
#include <iostream>
#include <GL/glut.h>

#include <stan/agrad/agrad.hpp>

using namespace simference;
using namespace std;
using namespace Eigen;

//typedef double RealNum;
typedef stan::agrad::var RealNum;

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
		auto dtree = Sample(*axiom);
		*derivedString = dtree.derivedString();
		if (mobile) delete mobile;
		mobile = new Mobile<RealNum>(*derivedString, *anchor);

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
	else if (key == 'a')
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