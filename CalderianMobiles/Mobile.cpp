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
				return ComponentPtr(new RodComponent(point, helper(leftPoint), helper(rightPoint)));
			}
			else throw "Mobile::Mobile - Malformed input string!";
		};

		std::reverse(derivation.symbols.begin(), derivation.symbols.end());
		root = helper(anchor);
	}

	void Mobile::render()
	{
		root->render();
	}

	// Render constants
	#define SET_STRING_COLOR (glColor3f(0.5f, 0.5f, 0.5f))
	#define SET_ROD_COLOR (glColor3f(0.2f, 0.2f, 0.2f))
	#define SET_WEIGHT_COLOR (glColor3f(0.2f, 0.2f, 1.0f))
	#define STRING_WIDTH 1.0f
	#define ROD_WIDTH 3.0f
	#define WEIGHT_SLICES 16

	// Simulation constants
	// TODO: Define these

	void Mobile::StringComponent::render()
	{
		glLineWidth(STRING_WIDTH);
		SET_STRING_COLOR;
		glMatrixMode(GL_MODELVIEW);
		glBegin(GL_LINES);
		glVertex2f(anchor.x(), anchor.y());
		glVertex2f(anchor.x(), anchor.y() - length);
		glEnd();

		child->render();
	}

	double Mobile::StringComponent::mass()
	{
		return 0.0;
	}

	void Mobile::WeightComponent::render()
	{
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		// Push down by the radius
		glTranslatef(anchor.x(), anchor.y()-radius, anchor.z());

		// Squish along z so we can render in '2d'
		glScalef(1.0f, 1.0f, 0.01f);

		// Draw a weight as a sphere
		SET_WEIGHT_COLOR;
		glutSolidSphere(radius, WEIGHT_SLICES, WEIGHT_SLICES);

		glPopMatrix();
	}

	double Mobile::WeightComponent::mass()
	{
		return 0.0;
	}

	void Mobile::RodComponent::render()
	{
		const Eigen::Vector3f& lp = leftChild->as<StringComponent>()->anchor;
		const Eigen::Vector3f& rp = rightChild->as<StringComponent>()->anchor;

		// Draw a rod
		glLineWidth(ROD_WIDTH);
		SET_ROD_COLOR;
		glMatrixMode(GL_MODELVIEW);
		glBegin(GL_LINES);
		glVertex2f(lp.x(), lp.y());
		glVertex2f(rp.x(), rp.y());
		glEnd();

		leftChild->render();
		rightChild->render();
	}

	double Mobile::RodComponent::mass()
	{
		return 0.0;
	}
}