
#include "constantMassForce.h"

extern void forceMassiqueConstanteKernel(
		const struct mappedParticlePointers *pt, const unsigned int nParticles, 
		const float mFx, const float mFy, const float mFz);

ConstantMassForce::ConstantMassForce(qglviewer::Vec Fm) :
	Fm(Fm)
{
}

ConstantMassForce::~ConstantMassForce() {
}

void ConstantMassForce::operator()(const ParticleGroup *particleGroup) {

	forceMassiqueConstanteKernel(
			particleGroup->getMappedRessources(), particleGroup->getParticleCount(), 
			Fm.x, Fm.y, Fm.z);
}
