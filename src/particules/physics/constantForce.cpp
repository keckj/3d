
#include "constantForce.h"

extern void forceConstanteKernel(
		const struct mappedParticlePointers *pt, const unsigned int nParticles, 
		const float Fx, const float Fy, const float Fz);

ConstantForce::ConstantForce(qglviewer::Vec f) :
	f(f)
{
}

ConstantForce::~ConstantForce() {
}

void ConstantForce::operator()(const ParticleGroup *particleGroup) {

	forceConstanteKernel(
			particleGroup->getMappedRessources(), particleGroup->getParticleCount(), 
			f.x, f.y, f.z);
}
