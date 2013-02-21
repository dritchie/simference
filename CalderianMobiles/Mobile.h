#ifndef __MOBILE_H
#define __MOBILE_H

#include "MobileGrammar.h"
#include "../Common/DAD.h"
#include "../Common/Math.h"
#include <stan/agrad/agrad.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <queue>
#include <iostream>
#include <memory>
#include <vector>
#include <GL/glut.h>

using namespace simference::Grammar;
using namespace simference::Grammar::MobileGrammar;

namespace simference
{
	template<typename RealNum>
	class Mobile
	{
	public:

		typedef Eigen::Matrix<RealNum, 3, 1> Vector3r;
		typedef unsigned long NodeNum;
		typedef BCAD<NodeNum> NodeCode;

		class Component
		{
		public:
			Component(NodeCode* parentCode, NodeNum siblingId, NodeNum numSiblings) :
				code(parentCode, siblingId, numSiblings) {}

			virtual void render() const = 0;
			virtual RealNum mass() const = 0;
			virtual void updateAnchors(const Vector3r& a) { anchor = a; }
			virtual unsigned int numChildren() const { return 0; }
			virtual Component* firstChild() const { return NULL; }
			virtual Component* secondChild() const { return NULL; }
			template<class T> bool is() { return dynamic_cast<T*>(this) != NULL; }
			template<class T> T* as() { return dynamic_cast<T*>(this); }
			bool isDescendantOf(Component* other) const { return code < other->code; }
			bool isAncestorOf(Component* other) const { return other->code < code; }

			Vector3r anchor;
			NodeCode code;
		};
		typedef std::shared_ptr<Component> ComponentPtr;

		class StringComponent : public Component
		{
		public:
			StringComponent(StringTerminal<RealNum>* st,
				NodeCode* parentCode, NodeNum siblingId, NodeNum numSiblings)
				: Component(parentCode, siblingId, numSiblings), sym(st) {}
			void render() const;
			RealNum mass() const;
			void updateAnchors(const Vector3r& a);
			unsigned int numChildren() const { return 1; }
			Component* firstChild() const { return child.get(); }
			StringTerminal<RealNum>* sym;
			ComponentPtr child;
		};

		class WeightComponent : public Component
		{
		public:
			WeightComponent(WeightTerminal<RealNum>* wt,
				NodeCode* parentCode, NodeNum siblingId, NodeNum numSiblings)
				: Component(parentCode, siblingId, numSiblings), sym(wt) {}
			void render() const;
			RealNum mass() const;
			RealNum collision(StringComponent* str) const;
			RealNum collision(WeightComponent* weight) const;
			WeightTerminal<RealNum>* sym;
		};

		class RodComponent : public Component
		{
		public:
			RodComponent(RodTerminal<RealNum>* rt,
				NodeCode* parentCode, NodeNum siblingId, NodeNum numSiblings)
				: Component(parentCode, siblingId, numSiblings), sym(rt) {}

			void render() const;
			RealNum mass() const;
			void updateAnchors(const Vector3r& a);
			Vector3r torque() const;
			unsigned int numChildren() const { return 2; }
			Component* firstChild() const { return leftChild.get(); }
			Component* secondChild() const { return rightChild.get(); }
			RealNum collision(RodComponent* rod) const;
			RealNum collision(StringComponent* str) const;
			RealNum collision(WeightComponent* weight) const;
			RealNum scaledConnectPoint() const
			{
				return sym->params[RodConnectPoint]*sym->params[RodLength];
			}
			ComponentPtr leftChild;
			ComponentPtr rightChild;
			RodTerminal<RealNum>* sym;
		};

		class CollisionSummary
		{
		public:
			CollisionSummary()
				: rodXrod(0.0), rodXstring(0.0), rodXweight(0.0), weightXstring(0.0), weightXweight(0.0),
				rodXrodN(0), rodXstringN(0), rodXweightN(0), weightXstringN(0), weightXweightN(0) {}
			void print() const;
			RealNum rodXrod, rodXstring, rodXweight,
				weightXstring, weightXweight;
			unsigned int rodXrodN, rodXstringN, rodXweightN,
				weightXstringN, weightXweightN;
		};

		Mobile(String<RealNum> derivation, const Eigen::Vector3d& anchor);

		void render() const;
		void updateAnchors(const Eigen::Vector3d& a)
		{
			rootAnchor = a;
			Vector3r ar(a.x(), a.y(), a.z());
			root->updateAnchors(ar);
		}
		void updateAnchors() { updateAnchors(rootAnchor); }
		CollisionSummary checkStaticCollisions() const;
		bool sanityCheckNodeCodes() const;
		void printNodeCodes() const;
		RealNum netTorqueNorm() const;
		RealNum softMaxTorqueNorm() const;

	private:
		ComponentPtr root;
		Eigen::Vector3d rootAnchor;
		static GLUquadric* quadric;

		template<class T> std::vector<T*> nodesOfType() const;
	};


	//////////////////// Implementation ///////////////////////////////


	template<typename RealNum>
	Mobile<RealNum>::Mobile(String<RealNum> derivation, const Eigen::Vector3d& anchor)
	{
		function<ComponentPtr(NodeCode*)> helper = [&helper, &derivation](NodeCode* code) -> ComponentPtr
		{
			SymbolPtr head = derivation.symbols.back();
			derivation.symbols.pop_back();
			if (head->is<StringTerminal<RealNum>>())
			{
				auto st = head->as<StringTerminal<RealNum>>();
				auto stringcomp = new StringComponent(st, code, st->index, 2);
				stringcomp->child = helper(&stringcomp->code);
				return ComponentPtr(stringcomp);
			}
			else if (head->is<WeightTerminal<RealNum>>())
			{
				return ComponentPtr(new WeightComponent(head->as<WeightTerminal<RealNum>>(), code, 0, 1));
			}
			else if (head->is<RodTerminal<RealNum>>())
			{
				auto rodcomp = new RodComponent(head->as<RodTerminal<RealNum>>(), code, 0, 1);
				rodcomp->leftChild = helper(&rodcomp->code);
				rodcomp->rightChild = helper(&rodcomp->code);
				return ComponentPtr(rodcomp);
			}
			else throw "Mobile::Mobile - Malformed input string!";
		};

		std::reverse(derivation.symbols.begin(), derivation.symbols.end());
		root = helper(NULL);
		updateAnchors(anchor);
	}

	template<typename RealNum>
	GLUquadric* Mobile<RealNum>::quadric = gluNewQuadric();

	template<typename RealNum>
	void Mobile<RealNum>::render() const
	{
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		// Squish along z so we can render in '2d'
		glScalef(1.0f, 1.0f, 0.01f);

		root->render();

		glPopMatrix();
	}

	template<typename RealNum>
	template<class T>
	std::vector<T*> Mobile<RealNum>::nodesOfType() const
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

	template<typename RealNum>
	typename Mobile<RealNum>::CollisionSummary Mobile<RealNum>::checkStaticCollisions() const
	{
		CollisionSummary summary;

		auto rods = nodesOfType<RodComponent>();
		auto strings = nodesOfType<StringComponent>();
		auto weights = nodesOfType<WeightComponent>();

		// rod vs. rod
		if (rods.size() > 0)
		{
			for (unsigned int i = 0; i < rods.size()-1; i++)
			{
				auto rod1 = rods[i];
				for (unsigned int j = i+1; j < rods.size(); j++)
				{
					auto rod2 = rods[j];
					// Compare against only non-descendants and non-ancestors
					if (!rod1->isDescendantOf(rod2) && !rod1->isAncestorOf(rod2))
					{
						RealNum c = rod1->collision(rod2);
						summary.rodXrod += c;
						summary.rodXrodN += (c > 0.0);
					}
				}
			}
		}

		// rod vs. string
		for (auto rod : rods) for (auto str : strings)
		{
			// Compare against only non-descendants and non-ancestors
			if (!str->isAncestorOf(rod) && !str->isDescendantOf(rod))
			{
				RealNum c = rod->collision(str);
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
				RealNum c = rod->collision(weight);
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
				RealNum c = weight->collision(str);
				summary.weightXstring += c;
				summary.weightXstringN += (c > 0.0);
			}
		}

		// weight vs. weight
		if (weights.size() > 0)
		{
			for (unsigned int i = 0; i < weights.size()-1; i++)
			{
				for (unsigned int j = i+1; j < weights.size(); j++)
				{
					// Have to check against every other weight, unfortunately
					RealNum c = weights[i]->collision(weights[j]);
					summary.weightXweight += c;
					summary.weightXweightN += (c > 0.0);
				}
			}
		}

		return summary;
	}

	template<typename RealNum> 
	bool Mobile<RealNum>::sanityCheckNodeCodes() const
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

	template<typename RealNum>
	void Mobile<RealNum>::printNodeCodes() const
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

	template<typename RealNum>
	RealNum Mobile<RealNum>::netTorqueNorm() const
	{
		RealNum accum = 0.0;
		auto rods = nodesOfType<RodComponent>();
		for (auto rod : rods)
		{
			accum += rod->torque().norm();
		}
		return accum / rods.size();
	}

	template<typename RealNum>
	RealNum Mobile<RealNum>::softMaxTorqueNorm() const
	{
		RealNum accum = 0.0;
		auto rods = nodesOfType<RodComponent>();
		std::vector<RealNum> torqueNorms(rods.size(), 0.0);
		for (unsigned int i = 0; i < rods.size(); i++)
		{
			torqueNorms[i] = rods[i]->torque().norm();
		}
		//RealNum hmax = *(std::max_element(torqueNorms.begin(), torqueNorms.end()));
		RealNum smax =  Math::softMax(torqueNorms, 0.25);
		return smax;
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
	#define WEIGHT_DENSITY 3.0
	#define GRAVITY Vector3r(0.0, -9.8, 0.0)

	template<typename RealNum>
	void Mobile<RealNum>::StringComponent::render() const
	{
		SET_STRING_COLOR;
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(anchor.x(), anchor.y(), anchor.z());
		glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
		gluCylinder(quadric, STRING_RADIUS, STRING_RADIUS, sym->params[StringLength], RADIAL_SLICES, 1);
		glPopMatrix();

		child->render();
	}
	// Specialization (see Mobile.cpp)
	template<> void Mobile<stan::agrad::var>::StringComponent::render() const;

	template<typename RealNum>
	RealNum Mobile<RealNum>::StringComponent::mass() const
	{
		// Volume: pi r^2 l
		return STRING_RADIUS*STRING_RADIUS * Math::Pi * sym->params[StringLength] * STRING_DENSITY
			+ child->mass();
	}

	template<typename RealNum>
	void Mobile<RealNum>::StringComponent::updateAnchors(const Vector3r& a)
	{
		Component::updateAnchors(a);
		child->updateAnchors(a - Vector3r(0.0, sym->params[StringLength], 0.0));
	}

	template<typename RealNum>
	void Mobile<RealNum>::WeightComponent::render() const
	{
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		RealNum radius = sym->params[WeightRadius];

		// Push down by the radius
		glTranslatef(anchor.x(), anchor.y()-radius, anchor.z());

		// Draw a weight as a sphere
		SET_WEIGHT_COLOR;
		glutSolidSphere(radius, RADIAL_SLICES, RADIAL_SLICES);
		glPopMatrix();
	}
	// Specialization (see Mobile.cpp)
	template<> void Mobile<stan::agrad::var>::WeightComponent::render() const;

	template<typename RealNum>
	RealNum Mobile<RealNum>::WeightComponent::mass() const
	{
		// Volume = 4/3 pi r^3
		RealNum radius = sym->params[WeightRadius];
		return 1.3333 * Math::Pi * radius*radius*radius * WEIGHT_DENSITY;
	}

	template<typename RealNum>
	RealNum Mobile<RealNum>::WeightComponent::collision(StringComponent* str) const
	{
		RealNum radius = sym->params[WeightRadius];
		RealNum length = str->sym->params[StringLength];

		// Collision = chord length
		Vector3r p = str->anchor; p.y() -= length;
		Vector3r center = anchor; center.y() -= radius;
		RealNum a = 1.0;
		RealNum b = 2*(p.y() - center.y());
		RealNum c = p.dot(p) - 2*p.dot(center) + center.dot(center) - radius*radius;
		RealNum r1, r2;
		int detsign = Math::solveQuadratic(a, b, c, r1, r2);
		if (detsign > 0)
		{
			RealNum ss = p.y();
			RealNum se = ss + length;
			RealNum ys = p.y() + r1;
			RealNum ye = p.y() + r2;
			return Math::intervalOverlapAmount(ss, se, ys, ye);
		}
		else return 0.0;
	}

	template<typename RealNum>
	RealNum Mobile<RealNum>::WeightComponent::collision(WeightComponent* weight) const
	{
		// Sanity check
		if (this == weight) return 0.0;

		RealNum radius1 = sym->params[WeightRadius];
		RealNum radius2 = weight->sym->params[WeightRadius];

		// collision = penetration distance
		Vector3r c1 = anchor; c1.y() -= radius1;
		Vector3r c2 = weight->anchor; c2.y() -= radius2;
		RealNum d = (c1-c2).norm();
		RealNum r = radius1 + radius2;
		return max(r-d, (RealNum)0.0);
	}

	template<typename RealNum>
	void Mobile<RealNum>::RodComponent::render() const
	{
		// Draw a rod
		SET_ROD_COLOR;
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(anchor.x()-scaledConnectPoint(), anchor.y(), anchor.z());
		glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
		gluCylinder(quadric, ROD_RADIUS, ROD_RADIUS, sym->params[RodLength], RADIAL_SLICES, 1);
		glPopMatrix();

		leftChild->render();
		rightChild->render();
	}
	// Specialization (see Mobile.cpp)
	template<> void Mobile<stan::agrad::var>::RodComponent::render() const;

	template<typename RealNum>
	RealNum Mobile<RealNum>::RodComponent::mass() const
	{
		// Volume: pi r^2 l
		return ROD_RADIUS*ROD_RADIUS * Math::Pi * sym->params[RodLength] * ROD_DENSITY
			+ leftChild->mass() + rightChild->mass();
	}

	template<typename RealNum>
	void Mobile<RealNum>::RodComponent::updateAnchors(const Vector3r& a)
	{
		Component::updateAnchors(a);
		Vector3r lp = a - Vector3r(scaledConnectPoint(), 0.0, 0.0);
		Vector3r rp = lp + Vector3r(sym->params[RodLength], 0.0, 0.0);
		leftChild->updateAnchors(lp);
		rightChild->updateAnchors(rp);
	}

	template<typename RealNum>
	typename Mobile<RealNum>::Vector3r Mobile<RealNum>::RodComponent::torque() const
	{
		Vector3r lp = anchor; lp.x() -= scaledConnectPoint();
		Vector3r rp = lp; rp.x() += sym->params[RodLength];
		Vector3r d1 = lp - anchor;
		Vector3r d2 = rp - anchor;
		Vector3r f1 = leftChild->mass() * GRAVITY;
		Vector3r f2 = rightChild->mass() * GRAVITY;
		return d1.cross(f1) + d2.cross(f2);
	}

	template<typename RealNum>
	RealNum Mobile<RealNum>::RodComponent::collision(RodComponent* rod) const
	{
		// Sanity check
		if (this == rod) return 0.0;

		// collision == overlap interval (linear)
		RealNum ymin1 = anchor.y() - ROD_RADIUS;
		RealNum ymax1 = anchor.y() + ROD_RADIUS;
		RealNum ymin2 = rod->anchor.y() - ROD_RADIUS;
		RealNum ymax2 = rod->anchor.y() + ROD_RADIUS;
		if (Math::intervalsOverlap(ymin1, ymax1, ymin2, ymax2))
		{
			RealNum xmin1 = anchor.x() - scaledConnectPoint();
			RealNum xmax1 = xmin1 + sym->params[RodLength];
			RealNum xmin2 = rod->anchor.x() - rod->scaledConnectPoint();
			RealNum xmax2 = xmin2 + rod->sym->params[RodLength];
			return Math::intervalOverlapAmount(xmin1, xmax1, xmin2, xmax2);
		}
		else return 0.0;
	}

	template<typename RealNum>
	RealNum Mobile<RealNum>::RodComponent::collision(StringComponent* str) const
	{
		// collision == min distance from intersection point
		// to any endpoint of either party (linear)
		RealNum x = str->anchor.x();
		RealNum rs = anchor.x() - scaledConnectPoint();
		RealNum re = rs + sym->params[RodLength];
		RealNum y = anchor.y();
		RealNum ss = str->anchor.y();
		RealNum se = str->anchor.y() - str->sym->params[StringLength];
		if ((x > rs && x < re) && (se < y && ss > y))
		{
			return min(x - rs, min(re - x, min(ss - y, y - se)));
		}
		else return 0.0;
	}

	template<typename RealNum>
	RealNum Mobile<RealNum>::RodComponent::collision(WeightComponent* weight) const
	{
		RealNum radius = weight->sym->params[WeightRadius];
		// collision == chord length (linear)
		Vector3r p = anchor; p.x() -= scaledConnectPoint();
		Vector3r center = weight->anchor; center.y() -= radius;
		RealNum a = 1.0;
		RealNum b = 2*(p.x() - center.x());
		RealNum c = p.dot(p) - 2*p.dot(center) + center.dot(center) - radius*radius;
		RealNum r1, r2;
		int detsign = Math::solveQuadratic(a, b, c, r1, r2);
		if (detsign > 0)
		{
			RealNum rs = p.x();
			RealNum re = rs + sym->params[RodLength];
			RealNum xs = p.x() + r1;
			RealNum xe = p.x() + r2;
			return Math::intervalOverlapAmount(rs, re, xs, xe);
		}
		else return 0.0;
	}

	template<typename RealNum>
	void Mobile<RealNum>::CollisionSummary::print() const
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

#endif