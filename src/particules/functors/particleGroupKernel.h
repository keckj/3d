#ifndef PARTICLEGROUPKERNEL_H
#define PARTICLEGROUPKERNEL_H

#include "particleGroup.h"

class ParticleGroup;

class ParticleGroupKernel {

	public:
		virtual ~ParticleGroupKernel() {};
		virtual void operator()(const ParticleGroup *particleGroup) = 0;
		virtual void animate() {};
		
	protected:
		ParticleGroupKernel() {};
};


#endif /* end of include guard: PARTICLEGROUPKERNEL_H */
