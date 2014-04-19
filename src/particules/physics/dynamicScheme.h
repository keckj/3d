
#ifndef DYNAMICSCHEME_H
#define DYNAMICSCHEME_H

#include "particleGroupKernel.h"

class DynamicScheme : public ParticleGroupKernel {

	public:
		DynamicScheme();
		~DynamicScheme();

		void operator ()(const ParticleGroup *particleGroup);
};


#endif /* end of include guard: DYNAMICSCHEME_H */
