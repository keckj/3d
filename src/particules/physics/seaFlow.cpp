

#include "seaFlow.h"

extern void forceConstanteKernel(
		const struct mappedParticlePointers *pt, const unsigned int nParticles, 
		const float Fx, const float Fy, const float Fz);

SeaFlow::SeaFlow(qglviewer::Vec flowDir) :
	flowDir(flowDir)
{
}

SeaFlow::~SeaFlow() {
}

void SeaFlow::operator()(const ParticleGroup *particleGroup) {
	
	static float dt = 0.01;
	static float t = 0.0f;
	qglviewer::Vec flow = 1*flowDir*sin(t);

	forceConstanteKernel(
			particleGroup->getMappedRessources(), particleGroup->getParticleCount(), 
			flow.x, flow.y, flow.z);
	
	t+= dt;
}
