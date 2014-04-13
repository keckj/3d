
#ifndef PARTICLEGROUP_H
#define PARTICLEGROUP_H

#include "consts.h"
#include "particule.h"
#include "program.h"
#include "renderTree.h"
#include <list>
#include <map>

struct mappedParticlePointers {
	float *x, *y, *z, *vx, *vy, *vz, *m, *im, *r, *kill;
};

class ParticleGroup : public RenderTree {

	public:
		ParticleGroup(unsigned int maxParticles);
		virtual ~ParticleGroup();
		
		void addParticle(Particule *p);
		void releaseParticles();

		virtual void drawDownwards(const float *modelMatrix = consts::identity4);
		virtual void animateDownwards();
		
		struct mappedParticlePointers getMappedRessources();

	private:
		unsigned int maxParticles;
		unsigned int nParticles;
		unsigned int nWaitingParticles;
		std::list<Particule *> particlesWaitList;
		
		int nBuffers;
		unsigned int *buffers; //VBOs
		unsigned int x_b, y_b, z_b, r_b, kill_b; //VBOs
		cudaGraphicsResource_t *ressources;
		cudaGraphicsResource_t x_r, y_r, z_r, r_r, kill_r;
		float *x_d, *y_d, *z_d, *vx_d, *vy_d, *vz_d, *m_d, *im_d, *r_d, *kill_d; //device pointers

		void fromDevice();
		void toDevice();

		void mapRessources();
		void unmapRessources();

		static Program *_debugProgram;
		static std::map<std::string, int> _uniformLocs;
		static void makeDebugProgram();
};

#endif /* end of include guard: PARTICLEGROUP_H */
