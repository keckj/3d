
#ifndef FROTTEMENTFLUIDEAVANCE_H
#define FROTTEMENTFLUIDEAVANCE_H


#include "particleGroupKernel.h"

// cf drag coefficient wikipedia
class FrottementFluideAvance : public ParticleGroupKernel {

	public:
		FrottementFluideAvance(float rho, 
				float cx, float cy, float cz);
		~FrottementFluideAvance();

		void operator ()(const ParticleGroup *particleGroup);

	private:
		float rho;
		float cx, cy, cz;
};



#endif /* end of include guard: FROTTEMENTFLUIDE_H */
