
#include "killParticles.h"

extern void killParticleKernel(const struct mappedParticlePointers *pt,
				const float vx, const float vy, const float vz,
				const unsigned int nParticles, const float maxVal);

KillParticles::KillParticles(qglviewer::Vec v, float maxVal) :
	maxVal(maxVal), vx(v.x), vy(v.y), vz(v.z)
{
}

KillParticles::~KillParticles() {
}

void KillParticles::operator()(const ParticleGroup *particleGroup) {

	killParticleKernel(
			particleGroup->getMappedRessources(), 
			vx, vy, vz, 
			particleGroup->getParticleCount(), 
			maxVal);
}
