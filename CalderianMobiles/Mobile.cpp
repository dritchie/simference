#include "Mobile.h"

using namespace std;
using namespace Eigen;
using namespace simference;
using namespace stan::agrad;

namespace simference
{
	template<>
	void Mobile<var>::StringComponent::render() const
	{
		SET_STRING_COLOR;
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(anchor.x().val(), anchor.y().val(), anchor.z().val());
		glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
		gluCylinder(quadric, STRING_RADIUS, STRING_RADIUS, sym->params[StringLength].val(), RADIAL_SLICES, 1);
		glPopMatrix();

		child->render();
	}

	template<>
	void Mobile<var>::WeightComponent::render() const
	{
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		var radius = sym->params[WeightRadius];

		// Push down by the radius
		glTranslatef(anchor.x().val(), (anchor.y()-radius).val(), anchor.z().val());

		// Draw a weight as a sphere
		SET_WEIGHT_COLOR;
		glutSolidSphere(radius.val(), RADIAL_SLICES, RADIAL_SLICES);
		glPopMatrix();
	}

	template<>
	void Mobile<var>::RodComponent::render() const
	{
		// Draw a rod
		SET_ROD_COLOR;
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef((anchor.x()-scaledConnectPoint()).val(), anchor.y().val(), anchor.z().val());
		glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
		gluCylinder(quadric, ROD_RADIUS, ROD_RADIUS, sym->params[RodLength].val(), RADIAL_SLICES, 1);
		glPopMatrix();

		leftChild->render();
		rightChild->render();
	}
}