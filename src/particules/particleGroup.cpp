
#include "headers.h"
#include "particleGroup.h"
#include "globals.h"
#include "kernelHeaders.h"

#define N_BUFFERS 8

Program *ParticleGroup::_particlesDebugProgram = 0;
Program *ParticleGroup::_springsDebugProgram = 0;
std::map<std::string, int> ParticleGroup::_particleUniformLocs;
std::map<std::string, int> ParticleGroup::_springsUniformLocs;

ParticleGroup::ParticleGroup(unsigned int maxParticles, unsigned int maxSprings) :
	maxParticles(maxParticles), nParticles(0), nWaitingParticles(0),
	maxSprings(maxSprings), nSprings(0), nWaitingSprings(0),

	x_b(0), y_b(0), z_b(0), 
	r_b(0), kill_b(0),
	springs_lines_b(0), springs_intensity_b(0),

	x_d(0), y_d(0), z_d(0), 
	vx_d(0), vy_d(0), vz_d(0), 
	fx_d(0), fy_d(0), fz_d(0), 
	m_d(0), im_d(0), r_d(0),

	springs_k_d(0), springs_Lo_d(0),
	springs_d_d(0), springs_Fmax_d(0), 
	springs_lines_d(0), springs_intensity_d(0),
	
	springs_id1_d(0), springs_id2_d(0), 

	kill_d(0), fixed_d(0), springs_kill_d(0),

	_mapped(false)
{

	//OPENGL MEMORY (WILL BE SHARED WITH CUDA)
	buffers = new unsigned int[N_BUFFERS];
	glGenBuffers(N_BUFFERS, buffers);

	//particles//
	x_b = buffers[0]; //float
	y_b = buffers[1]; //float
	z_b = buffers[2]; //float
	r_b = buffers[3]; //float
	kill_b = buffers[4]; //unsigned char

	for (int i = 0; i < 4; i++) {
		glBindBuffer(GL_ARRAY_BUFFER, buffers[i]);
		glBufferData(GL_ARRAY_BUFFER, maxParticles*sizeof(float), 0, GL_DYNAMIC_DRAW);
	}
	glBindBuffer(GL_ARRAY_BUFFER, buffers[4]);
	glBufferData(GL_ARRAY_BUFFER, maxParticles*sizeof(unsigned char), 0, GL_DYNAMIC_DRAW);

	//springs//
	springs_intensity_b = buffers[5]; //float
	springs_lines_b = buffers[6]; //float*6	(two points) XYZ X'Y'Z'
	springs_kill_b = buffers[7]; //unsigned char

	glBindBuffer(GL_ARRAY_BUFFER, buffers[5]);
	glBufferData(GL_ARRAY_BUFFER, 2*maxSprings*sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[6]);
	glBufferData(GL_ARRAY_BUFFER, 6*maxSprings*sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[7]);
	glBufferData(GL_ARRAY_BUFFER, maxSprings*sizeof(unsigned char), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	//CUDA MEMORY (CAN'T BE SHARED WITH OPENGL)
	//particles//
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &vx_d, maxParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &vy_d, maxParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &vz_d, maxParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &fx_d, maxParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &fy_d, maxParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &fz_d, maxParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &m_d, maxParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &im_d, maxParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &fixed_d, maxParticles*sizeof(unsigned char)));
	
	//springs//
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &springs_id1_d, maxSprings*sizeof(unsigned int)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &springs_id2_d, maxSprings*sizeof(unsigned int)));

	CHECK_CUDA_ERRORS(cudaMalloc((void**) &springs_k_d, maxSprings*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &springs_Lo_d, maxSprings*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &springs_d_d, maxSprings*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMalloc((void**) &springs_Fmax_d, maxSprings*sizeof(float)));

	
	//CREATE BINDINGS BETWEEN CUDA AND OPENGL
	ressources = new cudaGraphicsResource*[N_BUFFERS];
	CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterBuffer(&x_r, x_b, cudaGraphicsMapFlagsNone));
	CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterBuffer(&y_r, y_b, cudaGraphicsMapFlagsNone));
	CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterBuffer(&z_r, z_b, cudaGraphicsMapFlagsNone));
	CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterBuffer(&r_r, r_b, cudaGraphicsMapFlagsNone));
	CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterBuffer(&kill_r, kill_b, cudaGraphicsMapFlagsNone));
	ressources[0] = x_r;
	ressources[1] = y_r;
	ressources[2] = z_r;
	ressources[3] = r_r;
	ressources[4] = kill_r;

	CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterBuffer(&springs_lines_r, springs_lines_b, cudaGraphicsMapFlagsNone));
	CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterBuffer(&springs_intensity_r, springs_intensity_b, cudaGraphicsMapFlagsNone));
	CHECK_CUDA_ERRORS(cudaGraphicsGLRegisterBuffer(&springs_kill_r, springs_kill_b, cudaGraphicsMapFlagsNone));
	ressources[5] = springs_lines_r;
	ressources[6] = springs_intensity_r;
	ressources[7] = springs_kill_r;
}

ParticleGroup::~ParticleGroup() {

	//delete links
	for (int i = 0; i < N_BUFFERS; i++) {
		cudaGraphicsUnregisterResource(ressources[i]);
	}

	//openGL memory
	glDeleteBuffers(N_BUFFERS, buffers);

	//shared memory that has already been freed before
	//cudaFree(x_d); cudaFree(y_d); cudaFree(z_d); cudaFree(r_d); cudaFree(kill_d);
	//cudaFree(springs_lines_d); cudaFree(springs_intensity_d); cudaFree(springs_kill_d);

	//cuda memory
	CHECK_CUDA_ERRORS(cudaFree(vx_d));
	CHECK_CUDA_ERRORS(cudaFree(vy_d));
	CHECK_CUDA_ERRORS(cudaFree(vz_d));
	CHECK_CUDA_ERRORS(cudaFree(fx_d));
	CHECK_CUDA_ERRORS(cudaFree(fy_d));
	CHECK_CUDA_ERRORS(cudaFree(fz_d));
	CHECK_CUDA_ERRORS(cudaFree(m_d));
	CHECK_CUDA_ERRORS(cudaFree(im_d));
	CHECK_CUDA_ERRORS(cudaFree(r_d));
	CHECK_CUDA_ERRORS(cudaFree(fixed_d));
	
	CHECK_CUDA_ERRORS(cudaFree(springs_id1_d));
	CHECK_CUDA_ERRORS(cudaFree(springs_id2_d));
	CHECK_CUDA_ERRORS(cudaFree(springs_k_d));
	CHECK_CUDA_ERRORS(cudaFree(springs_Lo_d));
	CHECK_CUDA_ERRORS(cudaFree(springs_d_d));
	CHECK_CUDA_ERRORS(cudaFree(springs_Fmax_d));
	
	//cpu memory
	delete [] buffers;
	delete [] ressources;
		
	while(!particlesWaitList.empty()) delete particlesWaitList.front(), particlesWaitList.pop_front();
	while(!springsWaitList.empty()) delete springsWaitList.front(), springsWaitList.pop_front();
}


void ParticleGroup::addParticle(Particule *p) {
	if(nWaitingParticles >= maxParticles) {
		log_console.errorStream() 
			<< "Trying to add a particles but max number of particles is already at its max (" << maxParticles << ") !";
		exit(1);
	}
	else if(nWaitingParticles + nParticles >= maxParticles) {
		log_console.warnStream()
		<< "Waiting CPU particles count + GPU particles count > maxParticles (" << maxParticles << "), just hope they'll die !";
	}

	particlesWaitList.push_back(p);	
	nWaitingParticles++;
}

void ParticleGroup::addSpring(unsigned int particleId1, unsigned int particleId2, float k, float Lo, float d, float Fmax) {
	
	if(nWaitingSprings >= maxSprings) {
		log_console.errorStream() 
			<< "Trying to add a spring but max number of springs is already at its max (" << maxSprings << ") !";
		exit(1);
	}
	else if(particleId1 >= nWaitingParticles + nParticles
			|| particleId2 >= nWaitingParticles + nParticles) {
		log_console.errorStream() 
			<< "Trying to add a spring between particle " 
			<< particleId1 << " and " << particleId2 
			<< " but current simulated particles count is " 
			<< nParticles << " and there is only " << nWaitingParticles 
			<< " waiting to be transfered on GPU !";

		exit(1);
	}
	else if(nWaitingSprings + nSprings >= maxSprings) {
		log_console.warnStream()
		<< "Waiting CPU springs count + GPU springs count > maxSprings (" << maxSprings << "), just hope they'll die !";
	}

	Ressort *ressort = new Ressort(particleId1, particleId2, k, Lo, d, Fmax);
	springsWaitList.push_back(ressort);
	nWaitingSprings++;
}

void ParticleGroup::addKernel(ParticleGroupKernel *kernel) {
	kernels.push_back(kernel);
}
	
void ParticleGroup::drawDownwards(const float *modelMatrix) {

	static float *proj = new float[16], *view = new float[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, view);
    glGetFloatv(GL_PROJECTION_MATRIX, proj);

	if(_particlesDebugProgram == 0)
		makeDebugPrograms();
	
	_particlesDebugProgram->use();
	
	glPushAttrib(GL_COLOR_BUFFER_BIT);
        
    //glDisable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_POINT_SPRITE);
	glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
	
	glEnable(GL_LINE_SMOOTH);

	glEnable (GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glUniformMatrix4fv(_particleUniformLocs["projectionMatrix"], 1, GL_FALSE, proj);
	glUniformMatrix4fv(_particleUniformLocs["viewMatrix"], 1, GL_FALSE, view);
	glUniformMatrix4fv(_particleUniformLocs["modelMatrix"], 1, GL_TRUE, modelMatrix);
	
	glUniform1f(_particleUniformLocs["rmin"], 0.04);
	glUniform1f(_particleUniformLocs["rmax"], 0.2);

	glBindBuffer(GL_ARRAY_BUFFER, x_b);
	glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(0, 1);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, y_b);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(1, 1);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, z_b);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(2, 1);
	glEnableVertexAttribArray(2);
	
	glBindBuffer(GL_ARRAY_BUFFER, r_b);
	glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(3, 1);
	glEnableVertexAttribArray(3);

	glBindBuffer(GL_ARRAY_BUFFER, kill_b);
	glVertexAttribIPointer(4, 1, GL_UNSIGNED_BYTE, 0, 0);
	glVertexAttribDivisor(4, 1);
	glEnableVertexAttribArray(4);

	glDrawArraysInstanced(GL_POINTS, 0, 1, nParticles);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glUseProgram(0);
	
	_springsDebugProgram->use();
	glUniformMatrix4fv(_springsUniformLocs["projectionMatrix"], 1, GL_FALSE, proj);
	glUniformMatrix4fv(_springsUniformLocs["viewMatrix"], 1, GL_FALSE, view);
	glUniformMatrix4fv(_springsUniformLocs["modelMatrix"], 1, GL_TRUE, modelMatrix);
	
	glBindBuffer(GL_ARRAY_BUFFER, springs_lines_b);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(0, 0);
	glEnableVertexAttribArray(0);
	
	glBindBuffer(GL_ARRAY_BUFFER, springs_intensity_b);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(1, 0);
	glEnableVertexAttribArray(1);
	
    glLineWidth(1.0f);
	glDrawArrays(GL_LINES, 0,nSprings*2);

	glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SMOOTH);
	glDisable(GL_LINE_SMOOTH);
        //glEnable(GL_DEPTH_TEST);

	glPopAttrib();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glUseProgram(0);
}

void ParticleGroup::animateDownwards() {
	mapRessources();

	std::list<ParticleGroupKernel *>::iterator it = kernels.begin();
	for (; it != kernels.end(); ++it) {
		(*it)->animate();
		(**it)(this);
	}

	unmapRessources();
}

void ParticleGroup::fromDevice() {
	
	float *x_h=0, *y_h=0, *z_h=0, *vx_h=0, *vy_h=0, *vz_h=0, *m_h=0, *r_h=0; //8
	float *springs_k_h=0, *springs_Lo_h=0, *springs_d_h=0, *springs_Fmax_h=0; //4
	unsigned int *springs_id1_h=0, *springs_id2_h=0; //2
	unsigned char *kill_h=0, *fixed_h=0, *springs_kill_h=0; //3

	unsigned int nRessources = 17;
	void **data_p_h[17] = {
						(void**)&x_h, (void**)&y_h, (void**)&z_h, 
						(void**)&vx_h, (void**)&vy_h, (void**)&vz_h, (void**)&m_h, (void**)&r_h, //particles float
						(void**)&kill_h, (void**)&fixed_h, //particles unsigned char
						(void**)&springs_k_h, (void**)&springs_Lo_h, (void**)&springs_d_h, (void**)&springs_Fmax_h, //springs float
						(void**)&springs_id1_h, (void**)&springs_id2_h, //springs unsigned int
						(void**)&springs_kill_h //springs unsigned char
						};
	
	size_t fs = sizeof(float), bs = sizeof(unsigned char), is = sizeof(unsigned int);
	size_t ps = nParticles, ss = nSprings;
	size_t dataSize[17] = {
						ps*fs, ps*fs, ps*fs, ps*fs, ps*fs, ps*fs, ps*fs, ps*fs,
						ps*bs, ps*bs,
						ss*fs, ss*fs, ss*fs, ss*fs,
						ss*is, ss*is, 
						ss*bs
						};

	for (unsigned int i = 0; i < nRessources; i++) {
		CHECK_CUDA_ERRORS(cudaMallocHost(data_p_h[i], dataSize[i]));
	}
	
	mapRessources();
	{
		void *data_d[17] = {
						x_d, y_d, z_d, 
						vx_d, vy_d, vz_d, m_d, r_d, //particles float
						kill_d, fixed_d, //particles unsigned char
						springs_k_d, springs_Lo_d, springs_d_d, springs_Fmax_d, //springs float
						springs_id1_d, springs_id2_d, //springs unsigned int
						springs_kill_d //springs unsigned char
						};

		for (unsigned int i = 0; i < nRessources; i++) {
			CHECK_CUDA_ERRORS(cudaMemcpy(data_p_h[i][0], data_d[i], dataSize[i], cudaMemcpyDeviceToHost));
		}
	}
	unmapRessources();

	for (unsigned int i = 0; i < nParticles; i++) {
		if(!kill_h[i]) {
			particlesWaitList.push_back(new Particule(Vec(x_h[i], y_h[i], z_h[i]), Vec(vx_h[i], vy_h[i], vz_h[i]), m_h[i], r_h[i], fixed_h[i]));
			nWaitingParticles++;
		}
	}

	//TODO
	//for (unsigned int i = 0; i < nParticles; i++) {
		//if(!springs_kill_h[i]) {
			//springsWaitList.push_back(new Ressort());
			//nWaitingParticles++;
		//}
	//}

	for (unsigned int i = 0; i < nRessources; i++) {
		CHECK_CUDA_ERRORS(cudaFreeHost(*(data_p_h[i])));
	}
	
	nParticles = 0;
	nSprings = 0;
}

void ParticleGroup::toDevice() {
	

	//Alloc CPU arrays
	float *x_h=0, *y_h=0, *z_h=0, *vx_h=0, *vy_h=0, *vz_h=0, *m_h=0, *im_h, *r_h=0; //9
	unsigned char *fixed_h=0; //1
	float *springs_k_h=0, *springs_Lo_h=0, *springs_d_h=0, *springs_Fmax_h=0; //4
	unsigned int *springs_id1_h=0, *springs_id2_h=0; //2

	unsigned int nRessources = 16;
	void **data_p_h[16] = {
						(void**)&x_h, (void**)&y_h, (void**)&z_h, 
						(void**)&vx_h, (void**)&vy_h, (void**)&vz_h, 
						(void**)&m_h, (void**)&im_h, (void**)&r_h, //particles float
						(void**)&fixed_h, //particle unsigned chars
						(void**)&springs_k_h, (void**)&springs_Lo_h, 
						(void**)&springs_d_h, (void**)&springs_Fmax_h, //springs float
						(void**)&springs_id1_h, (void**)&springs_id2_h, //springs unsigned int
						};
	
	size_t fs = sizeof(float), is = sizeof(unsigned int), bs = sizeof(unsigned char);
	size_t ps = nWaitingParticles, ss = nWaitingSprings;
	size_t dataSize[16] = {
						ps*fs, ps*fs, ps*fs, ps*fs, ps*fs, ps*fs, ps*fs, ps*fs, ps*fs,
						ps*bs,
						ss*fs, ss*fs, ss*fs, ss*fs,
						ss*is, ss*is, 
						};
	
	for (unsigned int i = 0; i < nRessources; i++) {
		CHECK_CUDA_ERRORS(cudaMallocHost(data_p_h[i], dataSize[i]));
	}

	//AoS to SoA
	assert(nParticles == 0);
	assert(nSprings == 0);

	{
		std::list<Particule*>::iterator it = particlesWaitList.begin();
		int i = 0;
		for (; it != particlesWaitList.end(); it++) {
			Particule *p = *it;
			Vec pos = p->getPosition();	
			x_h[i] = pos.x;
			y_h[i] = pos.y;
			z_h[i] = pos.z;

			Vec vel = p->getVelocity();	
			vx_h[i] = vel.x;
			vy_h[i] = vel.y;
			vz_h[i] = vel.z;

			m_h[i] = p->getMass();
			im_h[i] = p->getInvMass();
			r_h[i] = p->getRadius();

			fixed_h[i] = p->isFixed();

			nParticles++;
			i++;
		}
	}

	{
		std::list<Ressort*>::iterator it = springsWaitList.begin();
		int i = 0;
		for (; it != springsWaitList.end(); it++) {
			Ressort *r = *it;
			springs_id1_h[i] = r->IdP1;
			springs_id2_h[i] = r->IdP2;
			springs_k_h[i] = r->k;
			springs_Lo_h[i] = r->Lo;
			springs_d_h[i] = r->d;
			springs_Fmax_h[i] = r->Fmax;

			nSprings++;
			i++;
		}
	}

	//on libère les ressources de la liste
	while(!particlesWaitList.empty()) delete particlesWaitList.front(), particlesWaitList.pop_front();
	while(!springsWaitList.empty()) delete springsWaitList.front(), springsWaitList.pop_front();
	nWaitingSprings = 0;
	nWaitingParticles = 0;

	//send to GPU
	mapRessources();
	{
		void *data_d[16] = {
						x_d, y_d, z_d, 
						vx_d, vy_d, vz_d, m_d, im_d, r_d, //particles float
						fixed_d, //particle unsigned chars
						springs_k_d, springs_Lo_d, springs_d_d, springs_Fmax_d, //springs float
						springs_id1_d, springs_id2_d, //springs unsigned int
						};

		for (unsigned int i = 0; i < nRessources; i++) {
			CHECK_CUDA_ERRORS(cudaMemcpy(data_d[i], data_p_h[i][0], dataSize[i], cudaMemcpyHostToDevice));
		}

		//set memory to 0
		CHECK_CUDA_ERRORS(cudaMemset(kill_d, 0, nParticles*sizeof(unsigned char)));
		CHECK_CUDA_ERRORS(cudaMemset(fx_d, 0, nParticles*sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMemset(fy_d, 0, nParticles*sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMemset(fz_d, 0, nParticles*sizeof(float)));
		
		CHECK_CUDA_ERRORS(cudaMemset(springs_lines_d, 0, 6*nSprings*sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMemset(springs_intensity_d, 0, nSprings*sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMemset(springs_kill_d, 0, nSprings*sizeof(unsigned char)));
	}
	unmapRessources();

	for (unsigned int i = 0; i < nRessources; i++) {
		CHECK_CUDA_ERRORS(cudaFreeHost(data_p_h[i][0]));
	}
}

void ParticleGroup::mapRessources() {
	CHECK_CUDA_ERRORS(cudaGraphicsMapResources(N_BUFFERS, ressources, 0));

	size_t size;
	CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &x_d, &size, x_r));	
	CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &y_d, &size, y_r));	
	CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &z_d, &size, z_r));	
	CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &r_d, &size, r_r));	
	CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &kill_d, &size, kill_r));	
	
	CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &springs_lines_d, &size, springs_lines_r));	
	CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &springs_intensity_d, &size, springs_intensity_r));	
	CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &springs_kill_d, &size, springs_kill_r));	

	_mapped = true;
}

void ParticleGroup::unmapRessources() {
	CHECK_CUDA_ERRORS(cudaGraphicsUnmapResources(N_BUFFERS, ressources, 0));

	x_d = 0;
	y_d = 0;
	z_d = 0;
	r_d = 0;
	kill_d = 0;

	springs_lines_d = 0;
	springs_intensity_d = 0;
	springs_kill_d = 0;

	_mapped = false;
}


void ParticleGroup::makeDebugPrograms() {

	_particlesDebugProgram = new Program("Particle");
	_particlesDebugProgram->bindAttribLocations("0 1 2 3 4", "x y z r alive");
	_particlesDebugProgram->bindFragDataLocation(0, "out_colour");

	_particlesDebugProgram->attachShader(Shader("shaders/particle/particle_vs.glsl", GL_VERTEX_SHADER));
	_particlesDebugProgram->attachShader(Shader("shaders/particle/particle_gs.glsl", GL_GEOMETRY_SHADER));
	_particlesDebugProgram->attachShader(Shader("shaders/particle/particle_fs.glsl", GL_FRAGMENT_SHADER));

	_particlesDebugProgram->link();
	_particleUniformLocs = _particlesDebugProgram->getUniformLocationsMap("modelMatrix projectionMatrix viewMatrix rmin rmax", true);
	
	_springsDebugProgram = new Program("Spring");
	_springsDebugProgram->bindAttribLocations("0 1 2", "pos intensity alive");
	_springsDebugProgram->bindFragDataLocation(0, "out_colour");

	_springsDebugProgram->attachShader(Shader("shaders/particle/spring_vs.glsl", GL_VERTEX_SHADER));
	_springsDebugProgram->attachShader(Shader("shaders/particle/spring_fs.glsl", GL_FRAGMENT_SHADER));
	
	_springsDebugProgram->link();
    _springsUniformLocs = _springsDebugProgram->getUniformLocationsMap("modelMatrix projectionMatrix viewMatrix", true);
}

void ParticleGroup::releaseParticles() {
	fromDevice();
	toDevice();
}

struct mappedParticlePointers *ParticleGroup::getMappedRessources() const {
	if(!_mapped) {
		log_console.errorStream() << "Trying to get ressources that have not been mapped !";	
		exit(1);
	}

	struct mappedParticlePointers *pt = new struct mappedParticlePointers;
	*pt = {
		x_d, y_d, z_d, vx_d, vy_d, vz_d, fx_d, fy_d, fz_d, m_d, im_d, r_d,
		kill_d, fixed_d,
		springs_k_d, springs_Lo_d, springs_d_d, springs_Fmax_d, springs_lines_d, springs_intensity_d,
		springs_id1_d, springs_id2_d, 
		springs_kill_d
	};

	return pt;
}


unsigned int ParticleGroup::getParticleCount() const {
	return nParticles;
}
unsigned int ParticleGroup::getMaxParticles() const {
	return maxParticles;
}
unsigned int ParticleGroup::getParticleWaitingCount() const {
	return nWaitingParticles;
}

unsigned int ParticleGroup::getSpringCount() const {
	return nSprings;
}
unsigned int ParticleGroup::getSpringWaitingCount() const {
	return nWaitingSprings;
}
unsigned int ParticleGroup::getMaxSprings() const {
	return maxSprings;
}
