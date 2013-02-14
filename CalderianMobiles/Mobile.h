#ifndef __MOBILE_H
#define __MOBILE_H

#include "../Common/DAD.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

// Forward declaration
class GLUquadric;

namespace simference
{
	// Forward declaration
	namespace Grammar
	{
		class String;
	}

	class Mobile
	{
	public:

		typedef unsigned long NodeNum;
		typedef BCAD<unsigned long> NodeCode;

		class Component
		{
		public:
			Component(NodeCode* parentCode, NodeNum siblingId, NodeNum numSiblings) :
				code(parentCode, siblingId, numSiblings) {}

			virtual void render() = 0;
			virtual double mass() = 0;
			virtual unsigned int numChildren() { return 0; }
			virtual Component* firstChild() { return NULL; }
			virtual Component* secondChild() { return NULL; }
			template<class T> bool is() { return dynamic_cast<T*>(this) != NULL; }
			template<class T> T* as() { return dynamic_cast<T*>(this); }
			bool isDescendantOf(Component* other) { return code < other->code; }
			bool isAncestorOf(Component* other) { return other->code < code; }

			NodeCode code;
		};
		typedef std::shared_ptr<Component> ComponentPtr;

		class StringComponent : public Component
		{
		public:
			StringComponent(const Eigen::Vector3f& a, double l,
				NodeCode* parentCode, NodeNum siblingId, NodeNum numSiblings)
				: Component(parentCode, siblingId, numSiblings), anchor(a), length(l) {}
			void render();
			double mass();
			unsigned int numChildren() { return 1; }
			Component* firstChild() { return child.get(); }
			Eigen::Vector3f anchor;
			double length;
			ComponentPtr child;
		};

		class WeightComponent : public Component
		{
		public:
			WeightComponent(const Eigen::Vector3f& a, double r,
				NodeCode* parentCode, NodeNum siblingId, NodeNum numSiblings)
				: Component(parentCode, siblingId, numSiblings), anchor(a), radius(r) {}

			void render();
			double mass();
			double collision(StringComponent* str);
			double collision(WeightComponent* weight);

			Eigen::Vector3f anchor;
			double radius;
		};

		class RodComponent : public Component
		{
		public:
			RodComponent(const Eigen::Vector3f& p, double l, double scp,
				NodeCode* parentCode, NodeNum siblingId, NodeNum numSiblings)
				: Component(parentCode, siblingId, numSiblings), pivot(p), length(l), stringConnectPoint(scp) {}

			void render();
			double mass();
			Eigen::Vector3f torque();
			unsigned int numChildren() { return 2; }
			Component* firstChild() { return leftChild.get(); }
			Component* secondChild() { return rightChild.get(); }
			double collision(RodComponent* rod);
			double collision(StringComponent* str);
			double collision(WeightComponent* weight);

			Eigen::Vector3f pivot;
			double length;
			double stringConnectPoint;
			ComponentPtr leftChild;
			ComponentPtr rightChild;
		};

		class CollisionSummary
		{
		public:
			CollisionSummary()
				: rodXrod(0.0), rodXstring(0.0), rodXweight(0.0), weightXstring(0.0), weightXweight(0.0),
			      rodXrodN(0), rodXstringN(0), rodXweightN(0), weightXstringN(0), weightXweightN(0) {}
			void print();
			double rodXrod, rodXstring, rodXweight,
				weightXstring, weightXweight;
			unsigned int rodXrodN, rodXstringN, rodXweightN,
				weightXstringN, weightXweightN;
		};


		Mobile(Grammar::String derivation, const Eigen::Vector3f& anchor);
		void render();
		CollisionSummary checkStaticCollisions();
		bool sanityCheckNodeCodes();
		void printNodeCodes();
		double netTorqueNorm();

	private:
		ComponentPtr root;
		static GLUquadric* quadric;

		template<class T> std::vector<T*> nodesOfType();
	};
}

#endif