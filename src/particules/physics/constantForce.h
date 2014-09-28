#ifndef CONSTANTFORCE_H
#define CONSTANTFORCE_H

#include "particleGroupKernel.h"

class ConstantForce : public ParticleGroupKernel {

	public:
		ConstantForce(qglviewer::Vec f);
		~ConstantForce();

		void operator ()(const ParticleGroup *particleGroup);

	private:
		qglviewer::Vec f;
};


#endif /* end of include guard: CONSTANTFORCE_H */
