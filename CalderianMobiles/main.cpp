#include "MobileGrammar.h"
#include <iostream>
#include <GL/glut.h>

using namespace simference;
using namespace simference::Grammar;
using namespace simference::Grammar::SimpleMobileGrammar;
using namespace std;

// Globals (yuck)
String axiom;
String derivedString;

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
	glTranslatef(0.0f, 9.5f, 0.0f);
	for (SymbolPtr s : derivedString.symbols)
	{
		Renderable* rt = dynamic_cast<Renderable*>(s.get());
		rt->render();
	}

	glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y)
{
	bool needsRedisplay = false;
	switch (key)
	{
	case 'n':
		auto dtree = Sample(axiom);
		derivedString = dtree.derivedString();
		//for (SymbolPtr s : derivedString.symbols)
		//	cout << s->print().c_str();
		//cout << endl;
		needsRedisplay = true;
		break;
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
	axiom.symbols.push_back(SymbolPtr(new StringTerminal(2.0)));
	axiom.symbols.push_back(SymbolPtr(new StringEndpointVariable(0)));

	glutMainLoop();

	return 0;
}