#ifndef __DAD_H
#define __DAD_H

#include "Math.h"
#include <cstdlib>

namespace simference
{
	// Implementation of the Determination of Ancestor-Descendant algorithm
	// See http://www.eng.auburn.edu/files/acad_depts/csse/csse_technical_reports/CSSE01-09.pdf

	// Has valid behavior for any unsigned integral type T.
	template<typename T>
	class BCAD
	{
	public:
		BCAD(BCAD* parentCode, T siblingId, T numSiblings)
		{
 			T part2length = (T)ceil(Math::log2((double)numSiblings));
			part2length = (part2length < 1 ? 1 : part2length);
			if (parentCode == NULL)
			{
				length = part2length;
				code = siblingId;
			}
			else
				// Concatenate the bits
			{
				length = parentCode->length + part2length;
				code = (parentCode->code << part2length) | siblingId;
			}
		}
		T code;
		T length;	// Actual code is stored in the lower 'length' bits of 'code.'

		// Is this a descendant of other?
		bool operator < (const BCAD& other) const
		{
			// Compare the first other.length bits of this.code
			// with other.code
			return other.code == (this->code >> (length - other.length));
		}

		void print() const
		{
			printf("Code: %u | length: %u", code, length);
		}
	};
}

#endif