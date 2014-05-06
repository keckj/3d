

#include "seaFlow.h"

extern void forceConstanteKernel(
		const struct mappedParticlePointers *pt, const unsigned int nParticles, 
		const float Fx, const float Fy, const float Fz);

SeaFlow::SeaFlow(qglviewer::Vec flowDir, float force, float deltaT) :
	flowDir(flowDir), force(force), deltaT(deltaT), t(0.0f)
{
}

SeaFlow::~SeaFlow() {
}

void SeaFlow::operator()(const ParticleGroup *particleGroup) {
	qglviewer::Vec flow = force*flowDir*sin(t);

	forceConstanteKernel(
			particleGroup->getMappedRessources(), particleGroup->getParticleCount(), 
			flow.x, flow.y, flow.z);
	
	t+= deltaT;
}
