

#include "frottementFluide.h"

extern void frottementFluideKernel(
		const struct mappedParticlePointers *pt, const unsigned int nParticles, 
		const float k1, const float k2);

FrottementFluide::FrottementFluide(float k1, float k2) :
	k1(k1), k2(k2)
{
}

FrottementFluide::~FrottementFluide() {
}

void FrottementFluide::operator()(const ParticleGroup *particleGroup) {

	frottementFluideKernel(
			particleGroup->getMappedRessources(), particleGroup->getParticleCount(), 
			k1, k2);
}
