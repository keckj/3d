

#ifndef ATTRACTORS_H
#define ATTRACTORS_H

#include "particleGroupKernel.h"

class Attractor : public ParticleGroupKernel {

	public:
		Attractor(float dMin, float dMax, float C);
		~Attractor();

		void operator ()(const ParticleGroup *particleGroup);

	private:
		float C;
		float dMin, dMax;
};

#endif /* end of include guard: ATTRACTORS_H */
