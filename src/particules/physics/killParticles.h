#ifndef KILLPARTICLES_H
#define KILLPARTICLES_H

#include "particleGroupKernel.h"

class KillParticles : public ParticleGroupKernel {

	public:
		KillParticles(qglviewer::Vec v, float maxVal);
		~KillParticles();

		void operator ()(const ParticleGroup *particleGroup);

	private:
		float maxVal;
		float vx, vy, vz;
};


#endif /* end of include guard: KILLPARTICLES_H */
