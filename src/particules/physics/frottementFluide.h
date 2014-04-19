
#ifndef FROTTEMENTFLUIDE_H
#define FROTTEMENTFLUIDE_H


#include "particleGroupKernel.h"

//F = -k1*v - k2*v^2
class FrottementFluide : public ParticleGroupKernel {

	public:
		FrottementFluide(float k1, float k2 = 0);
		~FrottementFluide();

		void operator ()(const ParticleGroup *particleGroup);

	private:
		float k1, k2;
};



#endif /* end of include guard: FROTTEMENTFLUIDE_H */
