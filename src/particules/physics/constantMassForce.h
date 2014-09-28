#ifndef CONSTANTMASSFORCE_H
#define CONSTANTMASSFORCE_H

#include "particleGroupKernel.h"


//F = m*f (f en N/kg)
class ConstantMassForce : public ParticleGroupKernel {

	public:
		ConstantMassForce(qglviewer::Vec Fm);
		~ConstantMassForce();

		void operator ()(const ParticleGroup *particleGroup);

	private:
		qglviewer::Vec Fm;
};


#endif /* end of include guard: CONSTANTFORCE_H */
