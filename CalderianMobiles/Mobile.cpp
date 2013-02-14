#include "Mobile.h"
#include "MobileGrammar.h"
#include <algorithm>
#include <GL/glut.h>

using namespace std;
using namespace Eigen;
using namespace simference::Grammar;
using namespace simference::Grammar::SimpleMobileGrammar;

namespace simference
{
	Mobile::Mobile(String derivation, const Vector3f& anchor)
	{
		function<ComponentPtr(const Vector3f&)> helper = [&helper, &derivation](const Vector3f& point) -> ComponentPtr
		{
			SymbolPtr head = derivation.symbols.back();
			derivation.symbols.pop_back();
			if (head->is<StringTerminal>())
			{
				double length = head->as<StringTerminal>()->length;
				return ComponentPtr(new StringComponent(point, length, helper(point - Vector3f(0.0f, length, 0.0f))));
			}
			else if (head->is<WeightTerminal>())
			{
				double radius = head->as<WeightTerminal>()->radius;
				return ComponentPtr(new WeightComponent(point, radius));
			}
			else if (head->is<RodTerminal>())
			{
				auto rt = head->as<RodTerminal>();
				Vector3f leftPoint = point - Vector3f(rt->stringConnectPoint, 0.0f, 0.0f);
				Vector3f rightPoint = point + Vector3f(rt->length - rt->stringConnectPoint, 0.0f, 0.0f);
				return ComponentPtr(new RodComponent(point, rt->length, helper(leftPoint), helper(rightPoint)));
			}
			else throw "Mobile::Mobile - Malformed input string!";
		};

		std::reverse(derivation.symbols.begin(), derivation.symbols.end());
		root = helper(anchor);
	}

	GLUquadric* Mobile::quadric = gluNewQuadric();

	void Mobile::render()
	{
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		// Squish along z so we can render in '2d'
		glScalef(1.0f, 1.0f, 0.01f);

		root->render();

		glPopMatrix();
	}

	Mobile::CollisionSummary Mobile::checkStaticCollisions()
	{
		CollisionSummary summary;

		// TODO: Fill in!

		return summary;
	}

	double Mobile::netTorqueNorm()
	{
		// TODO: Fill in!
		return 0.0;
	}

	// Rendering and simulation constants
	#define SET_STRING_COLOR (glColor3f(0.5f, 0.5f, 0.5f))
	#define SET_ROD_COLOR (glColor3f(0.2f, 0.2f, 0.2f))
	#define SET_WEIGHT_COLOR (glColor3f(0.2f, 0.2f, 1.0f))
	#define STRING_RADIUS 0.02
	#define ROD_RADIUS 0.05
	#define RADIAL_SLICES 16
	#define STRING_DENSITY 1.0
	#define ROD_DENSITY 2.0
	#define WEIGHT_DENSITY 4.0

	void Mobile::StringComponent::render()
	{
		SET_STRING_COLOR;
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(anchor.x(), anchor.y(), anchor.z());
		glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
		gluCylinder(quadric, STRING_RADIUS, STRING_RADIUS, length, RADIAL_SLICES, 1);
		glPopMatrix();

		child->render();
	}

	double Mobile::StringComponent::mass()
	{
		// TODO: Fill in!
		return 0.0;
	}

	void Mobile::WeightComponent::render()
	{
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		// Push down by the radius
		glTranslatef(anchor.x(), anchor.y()-radius, anchor.z());

		// Draw a weight as a sphere
		SET_WEIGHT_COLOR;
		glutSolidSphere(radius, RADIAL_SLICES, RADIAL_SLICES);
		glPopMatrix();
	}

	double Mobile::WeightComponent::mass()
	{
		// TODO: Fill in!
		return 0.0;
	}

	void Mobile::RodComponent::render()
	{
		const Eigen::Vector3f& lp = leftChild->as<StringComponent>()->anchor;

		// Draw a rod
		SET_ROD_COLOR;
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(lp.x(), lp.y(), lp.z());
		glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
		gluCylinder(quadric, ROD_RADIUS, ROD_RADIUS, length, RADIAL_SLICES, 1);
		glPopMatrix();

		leftChild->render();
		rightChild->render();
	}

	double Mobile::RodComponent::mass()
	{
		// TODO: Fill in!
		return 0.0;
	}
}