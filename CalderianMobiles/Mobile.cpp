#include "Mobile.h"
#include "MobileGrammar.h"
#include "../Common/Math.h"
#include <algorithm>
#include <queue>
#include <iostream>
#include <GL/glut.h>

using namespace std;
using namespace Eigen;
using namespace simference::Grammar;
using namespace simference::Grammar::SimpleMobileGrammar;

namespace simference
{
	Mobile::Mobile(String derivation, const Vector3f& anchor)
	{
		function<ComponentPtr(const Vector3f&, NodeCode*)> helper = [&helper, &derivation](const Vector3f& point, NodeCode* code) -> ComponentPtr
		{
			SymbolPtr head = derivation.symbols.back();
			derivation.symbols.pop_back();
			if (head->is<StringTerminal>())
			{
				auto st = head->as<StringTerminal>();
				double length = st->length;
				auto stringcomp = new StringComponent(point, length, code, st->index, 2);
				stringcomp->child = helper(point - Vector3f(0.0f, length, 0.0f), &stringcomp->code);
				return ComponentPtr(stringcomp);
			}
			else if (head->is<WeightTerminal>())
			{
				double radius = head->as<WeightTerminal>()->radius;
				return ComponentPtr(new WeightComponent(point, radius, code, 0, 1));
			}
			else if (head->is<RodTerminal>())
			{
				auto rt = head->as<RodTerminal>();
				Vector3f leftPoint = point - Vector3f(rt->stringConnectPoint, 0.0f, 0.0f);
				Vector3f rightPoint = point + Vector3f(rt->length - rt->stringConnectPoint, 0.0f, 0.0f);
				auto rodcomp = new RodComponent(point, rt->length, rt->stringConnectPoint, code, 0, 1);
				rodcomp->leftChild = helper(leftPoint, &rodcomp->code);
				rodcomp->rightChild = helper(rightPoint, &rodcomp->code);
				return ComponentPtr(rodcomp);
			}
			else throw "Mobile::Mobile - Malformed input string!";
		};

		std::reverse(derivation.symbols.begin(), derivation.symbols.end());
		root = helper(anchor, NULL);
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

	template<class T>
	vector<T*> Mobile::nodesOfType()
	{
		vector<T*> nodes;
		
		queue<Component*> fringe;
		fringe.push(root.get());
		while(!fringe.empty())
		{
			Component* top = fringe.front();
			fringe.pop();
			if (top->is<T>())
				nodes.push_back(top->as<T>());
			if (top->numChildren() > 0)
				fringe.push(top->firstChild());
			if (top->numChildren() > 1)
				fringe.push(top->secondChild());
		}

		return nodes;
	}

	Mobile::CollisionSummary Mobile::checkStaticCollisions()
	{
		CollisionSummary summary;

		auto rods = nodesOfType<RodComponent>();
		auto strings = nodesOfType<StringComponent>();
		auto weights = nodesOfType<WeightComponent>();

		// rod vs. rod
		for (unsigned int i = 0; i < rods.size()-1; i++)
		{
			auto rod1 = rods[i];
			for (unsigned int j = i+1; j < rods.size(); j++)
			{
				auto rod2 = rods[j];
				// Compare against only non-descendants and non-ancestors
				if (!rod1->isDescendantOf(rod2) && !rod1->isAncestorOf(rod2))
				{
					double c = rod1->collision(rod2);
					summary.rodXrod += c;
					summary.rodXrodN += (c > 0.0);
				}
			}
		}

		// rod vs. string
		for (auto rod : rods) for (auto str : strings)
		{
			// Compare against only non-descendants and non-ancestors
			if (!str->isAncestorOf(rod) && !str->isDescendantOf(rod))
			{
				double c = rod->collision(str);
				summary.rodXstring += c;
				summary.rodXstringN += (c > 0.0);
			}
		}

		// rod vs. weight
		for (auto rod : rods) for (auto weight : weights)
		{
			// Compare against only non-descendants
			if (!weight->isDescendantOf(rod))
			{
				double c = rod->collision(weight);
				summary.rodXweight += c;
				summary.rodXweightN += (c > 0.0);
			}
		}

		// weight vs. string
		for (auto weight : weights) for (auto str : strings)
		{
			// Compare against only non-ancestors
			if (!str->isAncestorOf(weight))
			{
				double c = weight->collision(str);
				summary.weightXstring += c;
				summary.weightXstringN += (c > 0.0);
			}
		}

		// weight vs. weight
		for (unsigned int i = 0; i < weights.size()-1; i++)
		{
			for (unsigned int j = i+1; j < weights.size(); j++)
			{
				// Have to check against every other weight, unfortunately
				double c = weights[i]->collision(weights[j]);
				summary.weightXweight += c;
				summary.weightXweightN += (c > 0.0);
			}
		}

		return summary;
	}

	bool Mobile::sanityCheckNodeCodes()
	{
		function<bool(Component*, vector<Component*>&)> helper =
			[&helper](Component* node, vector<Component*>& path) -> bool
		{
			for (auto c : path)
			{
				if (!c->isAncestorOf(node))
				{
					cout << "FAILED:" << endl;
					cout << "  "; c->code.print(); cout << endl;
					cout << "  does not register as an ancestor of" << endl;
					cout << "  "; node->code.print(); cout << endl;
					cout << "  but should." << endl;
					return false;
				}
				if (!node->isDescendantOf(c))
				{
					cout << "FAILED:" << endl;
					cout << "  "; node->code.print(); cout << endl;
					cout << "  does not register as a descendant of" << endl;
					cout << "  "; c->code.print(); cout << endl;
					cout << "  but should." << endl;
					return false;
				}
			}
			if (node->numChildren() > 0)
			{
				path.push_back(node);
				bool res = helper(node->firstChild(), path);
				if (!res) return false;
				if (node->numChildren() > 1)
					res &= helper(node->secondChild(), path);
				path.pop_back();
				return res;
			}
			return true;
		};

		vector<Component*> path;
		return helper(root.get(), path);
	}

	void Mobile::printNodeCodes()
	{
		function<void(Component*,int)> helper = [&helper](Component* node, int depth) -> void
		{
			for (int i = 0; i < depth; i++) printf("  ");
			node->code.print(); cout << endl;
			if (node->numChildren() > 0)
				helper(node->firstChild(), depth+1);
			if (node->numChildren() > 1)
				helper(node->secondChild(), depth+1);
		};

		helper(root.get(), 0);
	}

	double Mobile::netTorqueNorm()
	{
		// TODO: Fill in!
		return 0.0;
	}


//////////////////////////////////////////////////////////////////////////////////////////



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

	double Mobile::WeightComponent::collision(StringComponent* str)
	{
		// Collision = chord length
		Vector3f p = str->anchor; p.y() -= str->length;
		Vector3f center = anchor; center.y() -= radius;
		float a = 1.0f;
		float b = 2*(p.y() - center.y());
		float c = p.dot(p) - 2*p.dot(center) + center.dot(center) - radius*radius;
		float r1, r2;
		int detsign = Math::solveQuadratic(a, b, c, r1, r2);
		if (detsign > 0)
		{
			double ss = p.y();
			double se = ss + str->length;
			double ys = p.y() + r1;
			double ye = p.y() + r2;
			return Math::intervalOverlapAmount(ss, se, ys, ye);
		}
		else return 0.0;
	}

	double Mobile::WeightComponent::collision(WeightComponent* weight)
	{
		// Sanity check
		if (this == weight) return 0.0;

		// collision = penetration distance
		Vector3f c1 = anchor; c1.y() -= radius;
		Vector3f c2 = weight->anchor; c2.y() -= weight->radius;
		double d = (c1-c2).norm();
		double r = radius + weight->radius;
		return max(r-d, 0.0);
	}

	void Mobile::RodComponent::render()
	{
		// Draw a rod
		SET_ROD_COLOR;
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(pivot.x()-stringConnectPoint, pivot.y(), pivot.z());
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

	double Mobile::RodComponent::collision(RodComponent* rod)
	{
		// Sanity check
		if (this == rod) return 0.0;

		// collision == overlap interval (linear)
		double ymin1 = pivot.y() - ROD_RADIUS;
		double ymax1 = pivot.y() + ROD_RADIUS;
		double ymin2 = rod->pivot.y() - ROD_RADIUS;
		double ymax2 = rod->pivot.y() + ROD_RADIUS;
		if (Math::intervalsOverlap(ymin1, ymax1, ymin2, ymax2))
		{
			double xmin1 = pivot.x() - stringConnectPoint;
			double xmax1 = xmin1 + length;
			double xmin2 = rod->pivot.x() - rod->stringConnectPoint;
			double xmax2 = xmin2 + rod->length;
			return Math::intervalOverlapAmount(xmin1, xmax1, xmin2, xmax2);
		}
		else return 0.0;
	}

	double Mobile::RodComponent::collision(StringComponent* str)
	{
		// collision == min distance from intersection point
		// to any endpoint of either party (linear)
		double x = str->anchor.x();
		double rs = pivot.x() - stringConnectPoint;
		double re = rs + length;
		double y = pivot.y();
		double ss = str->anchor.y();
		double se = str->anchor.y() - str->length;
		if ((x > rs && x < re) && (se < y && ss > y))
		{
			return min(x - rs, min(re - x, min(ss - y, y - se)));
		}
		else return 0.0;
	}

	double Mobile::RodComponent::collision(WeightComponent* weight)
	{
		// collision == chord length (linear)
		Vector3f p = pivot; p.x() -= stringConnectPoint;
		Vector3f center = weight->anchor; center.y() -= weight->radius;
		float a = 1.0f;
		float b = 2*(p.x() - center.x());
		float c = p.dot(p) - 2*p.dot(center) + center.dot(center) - weight->radius*weight->radius;
		float r1, r2;
		int detsign = Math::solveQuadratic(a, b, c, r1, r2);
		if (detsign > 0)
		{
			double rs = p.x();
			double re = rs + length;
			double xs = p.x() + r1;
			double xe = p.x() + r2;
			return Math::intervalOverlapAmount(rs, re, xs, xe);
		}
		else return 0.0;
	}

	void Mobile::CollisionSummary::print()
	{
		cout << "Mobile static collision summary:" << endl;
		cout << "--------------------------------" << endl;
		cout << "   Rod vs.    Rod: " << rodXrodN << " | " << rodXrod << endl;
		cout << "   Rod vs. String: " << rodXstringN << " | " << rodXstring << endl;
		cout << "   Rod vs. Weight: " << rodXweightN << " | " << rodXweight << endl;
		cout << "Weight vs. String: " << weightXstringN << " | " << weightXstring << endl;
		cout << "Weight vs. Weight: " << weightXweightN << " | " << weightXweight << endl;
		cout << endl;
	}
}