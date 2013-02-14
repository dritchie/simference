#ifndef __MATH_H
#define __MATH_H

#include <cmath>

namespace simference
{
	namespace Math
	{
		static double Pi = 3.141592654;
		static double TwoPi = 2*Pi;

		template<typename T>
		T signum(T x)
		{
			T zero = (T)0.0;
			if (x == zero)
				return zero;
			else if (x < zero)
				return (T)-1.0;
			else if (x > zero)
				return (T)1.0;
			else return x;		// NaN
		}

		template<typename T>
		bool intervalsOverlap(T s1, T e1, T s2, T e2)
		{
			return s2 < e1 && s1 < e2;
		}

		template<typename T>
		T intervalOverlapAmount(T s1, T e1, T s2, T e2)
		{
			if (intervalsOverlap(s1, e1, s2, e2))
			{
				if (s1 < s2 && e1 < e2)
					return e1 - s2;
				else if (s2 < s1 && e2 < e1)
					return e2 - s1;
				else if (s1 < e2 && e2 < e1)
					return e2 - s2;
				else return e1 - s1;	// (s2 < s1 && e1 < e2)
			}
			else return (T)0;
		}

		// Return code indicates sign of the determinant
		template<typename T>
		int solveQuadratic(T a, T b, T c, T& root1, T& root2)
		{
			T det = b*b - 4*a*c;
			if (det < 0)
				return -1;
			T sdet = sqrt(det);
			root1 = (-b - sdet) / 2*a;
			root2 = (-b + sdet) / 2*a;
			if (det == 0)
				return 0;
			else return 1;
		}

		template<typename T>
		T log2(T val)
		{
			return log(val) / log((T)2.0);
		}
	}
}

#endif