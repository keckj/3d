
#ifndef ARCHIMEDE_H
#define ARCHIMEDE_H

#include "headers.h"
#include "particleGroupKernel.h"

class PousseeArchimede : public ParticleGroupKernel {

	public:
		PousseeArchimede(qglviewer::Vec g, float rho);
		~PousseeArchimede();

		void operator ()(const ParticleGroup *particleGroup);

	private:
		qglviewer::Vec g;
		float rho;
};

#endif /* end of include guard: BULLES_H */
