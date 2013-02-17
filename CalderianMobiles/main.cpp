#include "MobileGrammar.h"
#include "Mobile.h"
#include <iostream>
#include <GL/glut.h>

using namespace simference;
using namespace simference::Grammar;
using namespace simference::Grammar::SimpleMobileGrammar;
using namespace std;
using namespace Eigen;

// Globals (yuck)
String axiom;
String derivedString;
Mobile* mobile = NULL;
Vector3f anchor(0.0f, 9.5f, 0.0f);

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
		auto dtree = Sample(axiom);
		derivedString = dtree.derivedString();
		if (mobile) delete mobile;
		mobile = new Mobile(derivedString, anchor);

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

	// We start with a single string (the string from which everything hangs)
	auto root = new StringTerminal(0);
	root->params[0] = 2.0;
	axiom.symbols.push_back(SymbolPtr(root));
	axiom.symbols.push_back(SymbolPtr(new StringEndpointVariable(0)));

	glutMainLoop();

	return 0;
}