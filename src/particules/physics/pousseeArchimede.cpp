#include "pousseeArchimede.h"
#include "kernelHeaders.h"


extern void pousseeArchimedeKernel(
		const struct mappedParticlePointers *pt, unsigned int nParticles, 
		float nx, float ny, float nz, 
		float rho, float g);

PousseeArchimede::PousseeArchimede(qglviewer::Vec g, float rho) :
	g(g), rho(rho)
{
}

PousseeArchimede::~PousseeArchimede() {
}

void PousseeArchimede::operator()(const ParticleGroup *particleGroup) {
	
	float gn = g.norm();
	qglviewer::Vec gv = g.unit();

	pousseeArchimedeKernel(particleGroup->getMappedRessources(), particleGroup->getParticleCount(), 
			gv.x, gv.y, gv.z,
			rho, gn);
}
