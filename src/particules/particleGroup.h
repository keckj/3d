
#ifndef PARTICLEGROUP_H
#define PARTICLEGROUP_H

#include "consts.h"
#include "particule.h"
#include "program.h"
#include "ressort.h"
#include "renderTree.h"
#include "particleGroupKernel.h"
#include <list>
#include <map>

struct mappedParticlePointers {
	//particules
	float *x, *y, *z, *vx, *vy, *vz, *fx, *fy, *fz, *m, *im, *r;
	bool *kill, *fixed;
	//ressorts
	float *k, *Lo, *d, *Fmax, *lines, *intensity;
	unsigned int *id1, *id2;
	bool *killSpring;
};

class ParticleGroupKernel;

class ParticleGroup : public RenderTree {

	public:
		ParticleGroup(unsigned int maxParticles, unsigned int maxSprings);
		virtual ~ParticleGroup();

		unsigned int getParticleCount() const;
		unsigned int getParticleWaitingCount() const;
		unsigned int getMaxParticles() const;
		
		unsigned int getSpringCount() const;
		unsigned int getSpringWaitingCount() const;
		unsigned int getMaxSprings() const;
		
		void addParticle(Particule *p);
		void addSpring(unsigned int particleId_1, unsigned int particleId_2, float k, float Lo, float d, float Fmax=-1.0f);
		void addKernel(ParticleGroupKernel *kernel);
	
		void releaseParticles();

		virtual void drawDownwards(const float *modelMatrix = consts::identity4);
		virtual void animateDownwards();
		
		struct mappedParticlePointers *getMappedRessources() const;

	private:
		unsigned int maxParticles, nParticles, nWaitingParticles;
		unsigned int maxSprings, nSprings, nWaitingSprings;
		
		std::list<Particule *> particlesWaitList;
		std::list<Ressort *> springsWaitList;

		std::list<ParticleGroupKernel *> kernels;
		
		//VBOs 
		unsigned int *buffers;
		unsigned int x_b, y_b, z_b,
					 r_b, kill_b, 
					 springs_lines_b, springs_intensity_b, springs_kill_b; //GL_LINES, FOR COLOR/THIKNESS, KILL 
		
		//graphic ressources to share context
		cudaGraphicsResource_t *ressources;
		cudaGraphicsResource_t x_r, y_r, z_r, 
        				   r_r, kill_r,
	        			   springs_lines_r, springs_intensity_r, springs_kill_r;
	
		//device pointers
		float *x_d, *y_d, *z_d, 
			  *vx_d, *vy_d, *vz_d, 
			  *fx_d, *fy_d, *fz_d,
			  *m_d, *im_d, *r_d,
			  *springs_k_d, *springs_Lo_d, *springs_d_d, *springs_Fmax_d, 
			  *springs_lines_d, *springs_intensity_d;

		unsigned int *springs_id1_d, *springs_id2_d;
		bool *kill_d, *fixed_d, *springs_kill_d;
		
		//funcs
		bool _mapped;

		void fromDevice();
		void toDevice();

		void mapRessources();
		void unmapRessources();

		static Program *_particlesDebugProgram, *_springsDebugProgram;
		static std::map<std::string, int> _particleUniformLocs, _springsUniformLocs;
		static void makeDebugPrograms();
};

#endif /* end of include guard: PARTICLEGROUP_H */
