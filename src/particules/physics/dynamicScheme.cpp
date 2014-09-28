
#include "dynamicScheme.h"
#include "kernelHeaders.h"

extern void dynamicSchemeKernel(const struct mappedParticlePointers *pt, unsigned int nParticles);

DynamicScheme::DynamicScheme() {
}

DynamicScheme::~DynamicScheme() {
}

void DynamicScheme::operator()(const ParticleGroup *particleGroup) {
	dynamicSchemeKernel(particleGroup->getMappedRessources(), particleGroup->getParticleCount());
}
