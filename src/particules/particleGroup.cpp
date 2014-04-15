
#include "headers.h"
#include "particleGroup.h"
#include "globals.h"
#include "kernelHeaders.h"

#define N_BUFFERS 8

Program *ParticleGroup::_debugProgram = 0;
std::map<std::string, int> ParticleGroup::_uniformLocs;

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

	kill_d(0), springs_kill_d(0),

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
	kill_b = buffers[4]; //bool

	for (int i = 0; i < 4; i++) {
		glBindBuffer(GL_ARRAY_BUFFER, buffers[i]);
		glBufferData(GL_ARRAY_BUFFER, maxParticles*sizeof(float), 0, GL_DYNAMIC_DRAW);
	}
	glBindBuffer(GL_ARRAY_BUFFER, buffers[4]);
	glBufferData(GL_ARRAY_BUFFER, maxParticles*sizeof(bool), 0, GL_DYNAMIC_DRAW);

	//springs//
	springs_intensity_b = buffers[5]; //float
	springs_lines_b = buffers[6]; //float*6	(two points) XYZ X'Y'Z'
	springs_kill_b = buffers[7]; //bool

	glBindBuffer(GL_ARRAY_BUFFER, buffers[5]);
	glBufferData(GL_ARRAY_BUFFER, maxSprings*sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[6]);
	glBufferData(GL_ARRAY_BUFFER, 6*maxSprings*sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, buffers[7]);
	glBufferData(GL_ARRAY_BUFFER, maxSprings*sizeof(bool), 0, GL_DYNAMIC_DRAW);
	
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
}

void ParticleGroup::addKernel(ParticleGroupKernel *kernel) {
	kernels.push_back(kernel);
}
		
void ParticleGroup::drawDownwards(const float *modelMatrix) {

	static float *proj = new float[16], *view = new float[16];

	if(_debugProgram == 0)
		makeDebugProgram();
	
	_debugProgram->use();

	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SMOOTH);

	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glUniformMatrix4fv(_uniformLocs["projectionMatrix"], 1, GL_FALSE, proj);
	glUniformMatrix4fv(_uniformLocs["viewMatrix"], 1, GL_FALSE, view);
	glUniformMatrix4fv(_uniformLocs["modelMatrix"], 1, GL_TRUE, modelMatrix);

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
	glVertexAttribPointer(4, 1, GL_BYTE, GL_FALSE, 0, 0);
	glVertexAttribDivisor(4, 1);
	glEnableVertexAttribArray(4);

	glDrawArraysInstanced(GL_POINTS, 0, 1, nParticles);
	
	glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SMOOTH);

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
	
	float *x_h=0, *y_h=0, *z_h=0, *vx_h=0, *vy_h=0, *vz_h=0, *m_h=0, *r_h=0;
	float *springs_k_h=0, *springs_Lo_h=0, *springs_d_h=0, *springs_Fmax_h=0;
	unsigned int *springs_id1_h, *springs_id2_h;
	bool *kill_h=0, *springs_kill_h=0;

	//particles
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&x_h), nParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&y_h), nParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&z_h), nParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&vx_h), nParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&vy_h), nParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&vz_h), nParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&m_h), nParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&r_h), nParticles*sizeof(float)));
	
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&kill_h), nParticles*sizeof(bool)));

	//springs
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&springs_k_d), nSprings*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&springs_Lo_d), nSprings*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&springs_d_d), nSprings*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&springs_Fmax_d), nSprings*sizeof(float)));

	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&springs_id1_d), nSprings*sizeof(unsigned int)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&springs_id2_d), nSprings*sizeof(unsigned int)));
	
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&springs_kill_d), nSprings*sizeof(bool)));
	
	float *data_h[] = {x_h, y_h, z_h, vx_h, vy_h, vz_h, m_h, r_h};
	int nData = 8;
	
	mapRessources();
	{
		float *data_d[] = {x_d, y_d, z_d, vx_d, vy_d, vz_d, m_d, r_d};

		for (int i = 0; i < nData - 1; i++) {
			CHECK_CUDA_ERRORS(cudaMemcpy(data_h[i], data_d[i], nParticles*sizeof(float), cudaMemcpyDeviceToHost));
		}
		CHECK_CUDA_ERRORS(cudaMemcpy(kill_h, kill_d, nParticles*sizeof(bool), cudaMemcpyDeviceToHost));
	}
	unmapRessources();

	for (unsigned int i = 0; i < nParticles; i++) {
		if(!kill_h[i]) {
			particlesWaitList.push_back(new Particule(Vec(x_h[i], y_h[i], z_h[i]), Vec(vx_h[i], vy_h[i], vz_h[i]), m_h[i], r_h[i]));
			nWaitingParticles++;
		}
	}

	for (int i = 0; i < nData; i++) {
		CHECK_CUDA_ERRORS(cudaFreeHost(data_h[i]));
	}
	CHECK_CUDA_ERRORS(cudaFreeHost(kill_h));
	nParticles = 0;
}

void ParticleGroup::toDevice() {
	float *x_h=0, *y_h=0, *z_h=0, *vx_h=0, *vy_h=0, *vz_h=0, *m_h=0, *im_h=0, *r_h=0;

	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&x_h), nWaitingParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&y_h), nWaitingParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&z_h), nWaitingParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&vx_h), nWaitingParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&vy_h), nWaitingParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&vz_h), nWaitingParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&m_h), nWaitingParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&im_h), nWaitingParticles*sizeof(float)));
	CHECK_CUDA_ERRORS(cudaMallocHost((void**) (&r_h), nWaitingParticles*sizeof(float)));


	std::list<Particule*>::iterator it = particlesWaitList.begin();
	int i = 0;
	nParticles = 0;
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

		nParticles++;
		i++;
	}

	particlesWaitList.clear();
	nWaitingParticles = 0;
	
	float *data_h[] = {x_h, y_h, z_h, vx_h, vy_h, vz_h, m_h, im_h, r_h};

	mapRessources();
	{
		float *data_d[] = {x_d, y_d, z_d, vx_d, vy_d, vz_d, m_d, im_d, r_d};

		for (unsigned int i = 0; i < 9; i++) {
			CHECK_CUDA_ERRORS(cudaMemcpy(data_d[i], data_h[i], nParticles*sizeof(float), cudaMemcpyHostToDevice));
		}

		CHECK_CUDA_ERRORS(cudaMemset(kill_d, 0, nParticles*sizeof(bool)));
		CHECK_CUDA_ERRORS(cudaMemset(fx_d, 0, nParticles*sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMemset(fy_d, 0, nParticles*sizeof(float)));
		CHECK_CUDA_ERRORS(cudaMemset(fz_d, 0, nParticles*sizeof(float)));
	}
	unmapRessources();
		
	for (unsigned int i = 0; i < 9; i++) {
			CHECK_CUDA_ERRORS(cudaFreeHost(data_h[i]));
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
	
	CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &springs_lines_b, &size, springs_lines_r));	
	CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &springs_intensity_b, &size, springs_intensity_r));	
	CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**) &springs_kill_b, &size, springs_kill_r));	

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


void ParticleGroup::makeDebugProgram() {
		_debugProgram = new Program("ParticleGroup Debug Program");

        _debugProgram->bindAttribLocations("0 1 2 3 4", "x y z r alive");
        _debugProgram->bindFragDataLocation(0, "out_colour");

        _debugProgram->attachShader(Shader("shaders/particle/vs.glsl", GL_VERTEX_SHADER));
        _debugProgram->attachShader(Shader("shaders/particle/fs.glsl", GL_FRAGMENT_SHADER));

        _debugProgram->link();
		
       _uniformLocs = _debugProgram->getUniformLocationsMap("modelMatrix projectionMatrix viewMatrix", true);
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
	*pt = {x_d, y_d, z_d, vx_d, vy_d, vz_d, fx_d, fy_d, fz_d, m_d, im_d, r_d,
		kill_d,
		springs_k_d, springs_Lo_d, springs_d_d, springs_Fmax_d,
		springs_id1_d, springs_id2_d, 
		springs_kill_d};

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
