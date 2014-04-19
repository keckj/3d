
#ifndef SEAFLOW_H
#define SEAFLOW_H

#include "particleGroupKernel.h"

class SeaFlow : public ParticleGroupKernel {

	public:
		SeaFlow(qglviewer::Vec flowDir);
		~SeaFlow();

		void operator ()(const ParticleGroup *particleGroup);

	private:
		qglviewer::Vec flowDir;
};

#endif /* end of include guard: SEAFLOW_H */
