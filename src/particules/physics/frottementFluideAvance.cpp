

#include "frottementFluideAvance.h"

extern void frottementFluideAvanceKernel(
		const struct mappedParticlePointers *pt, const unsigned int nParticles, 
		const float rho, 
		const float cx, const float cy, const float cz);

FrottementFluideAvance::FrottementFluideAvance(float rho, float cx, float cy, float cz) :
	rho(rho), cx(cx), cy(cy), cz(cz)
{
}

FrottementFluideAvance::~FrottementFluideAvance() {
}

void FrottementFluideAvance::operator()(const ParticleGroup *particleGroup) {

	frottementFluideAvanceKernel(
			particleGroup->getMappedRessources(), particleGroup->getParticleCount(), 
			rho, cx, cy, cz);
}
