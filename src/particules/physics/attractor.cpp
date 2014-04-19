
#include "attractor.h"

extern void attractorKernel(const struct mappedParticlePointers *pt, 
		const unsigned int nParticles,
		const float dMin, const float dMax, const float C);

Attractor::Attractor(float dMin, float dMax, float C) :
	C(C), dMin(dMin), dMax(dMax)
{
}

Attractor::~Attractor() {
}

void Attractor::operator()(const ParticleGroup *particleGroup) {
	attractorKernel(
			particleGroup->getMappedRessources(), particleGroup->getParticleCount(), 
			dMin, dMax, C);
}
