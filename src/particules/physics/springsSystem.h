#ifndef SPRINGSSYSTEM_H
#define SPRINGSSYSTEM_H

#include "particleGroupKernel.h"

class SpringsSystem : public ParticleGroupKernel {

	public:
		SpringsSystem(bool dumping);
		~SpringsSystem();

		void operator ()(const ParticleGroup *particleGroup);

	private:
		bool dumping;
};

#endif /* end of include guard: SPRINGSSYSTEM_H */
