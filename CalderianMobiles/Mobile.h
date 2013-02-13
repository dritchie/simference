#ifndef __MOBILE_H
#define __MOBILE_H

#include <Eigen/Core>
#include <memory>

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
		Mobile(Grammar::String derivation, const Eigen::Vector3f& anchor);
		void render();

		class Component
		{
		public:
			virtual void render() = 0;
			virtual double mass() = 0;
			template<class T> bool is() { return dynamic_cast<T*>(this) != NULL; }
			template<class T> T* as() { return dynamic_cast<T*>(this); }
		};
		typedef std::shared_ptr<Component> ComponentPtr;

		class StringComponent : public Component
		{
		public:
			StringComponent(const Eigen::Vector3f& a, double l, ComponentPtr c)
				: anchor(a), length(l), child(c) {}
			void render();
			double mass();
			Eigen::Vector3f anchor;
			double length;
			ComponentPtr child;
		};

		class WeightComponent : public Component
		{
		public:
			WeightComponent(const Eigen::Vector3f& a, double r) : anchor(a), radius(r) {}
			void render();
			double mass();
			Eigen::Vector3f anchor;
			double radius;
		};

		class RodComponent : public Component
		{
		public:
			RodComponent(const Eigen::Vector3f& p, ComponentPtr lc, ComponentPtr rc)
				: pivot(p), leftChild(lc), rightChild(rc) {}
			void render();
			double mass();
			Eigen::Vector3f pivot;
			ComponentPtr leftChild;
			ComponentPtr rightChild;
		};

	private:
		ComponentPtr root;
	};
}

#endif