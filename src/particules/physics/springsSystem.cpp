
#include "springsSystem.h"

extern void springKernel(
		const struct mappedParticlePointers *pt,
		const unsigned int nSprings,
		const bool handleDumping);

SpringsSystem::SpringsSystem(bool dumping) :
	dumping(dumping)
{
}

SpringsSystem::~SpringsSystem() {
}

void SpringsSystem::operator()(const ParticleGroup *particleGroup) {

	springKernel(
			particleGroup->getMappedRessources(), 
			particleGroup->getSpringCount(), 
			dumping);
}
