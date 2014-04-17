

#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"

#include <iostream>

extern void checkKernelExecution();

struct mappedParticlePointers {
	//particules
	float *x, *y, *z, *vx, *vy, *vz, *fx, *fy, *fz, *m, *im, *r;
	bool *kill;
	//ressorts
	float *k, *Lo, *d, *Fmax, *lines, *intensity;
	unsigned int *id1, *id2;
	bool *killSpring;
};

__global__ void forceConstante(
		float *fx, float *fy, float *fz,
		const int nParticles, 
		const float Fx, const float Fy, const float Fz) {


	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id >= nParticles)
		return;

	fx[id] += Fx;
	fy[id] += Fy;
	fz[id] += Fz;
}

__global__ void forceMassiqueConstante(
		float *fx, float *fy, float *fz,
		float *m, 
		const int nParticles, 
		const float mFx, const float mFy, const float mFz) {

	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id >= nParticles)
		return;

	float _m = m[id];

	fx[id] += _m*mFx;
	fy[id] += _m*mFy;
	fz[id] += _m*mFz;
}


__global__ void pousseeArchimede(float *x, float *y, float *z, 
		float *fx, float *fy, float *fz,
		float *r,  
		const int nParticles, 
		const float nx, const float ny, const float nz,
		const float rho, const float g) {


	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id >= nParticles)
		return;

	float _r = r[id];
	float V = 4/3.0*3.14*_r*_r*_r; 

	fx[id] += -nx*rho*V*g;
	fy[id] += -ny*rho*V*g;
	fz[id] += -nz*rho*V*g;
}

__global__ void frottementFluide(
		float *x, float *y, float *z, 
		float *vx, float *vy, float *vz,
		float *fx, float *fy, float *fz,
		const int nParticles, 
		const float k1, const float k2) {
	
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id >= nParticles)
		return;

	float _vx = vx[id], _vy = vy[id], _vz = vz[id];

	fx[id] += -(k1*_vx + k2*_vx*_vx);
	fy[id] += -(k1*_vy + k2*_vy*_vy);
	fz[id] += -(k1*_vz + k2*_vz*_vz);
}

__global__ void frottementFluideAvance(
		float *x, float *y, float *z, 
		float *vx, float *vy, float *vz,
		float *fx, float *fy, float *fz,
		float *r,
		const int nParticles, 
		const float rho, 
		const float cx, const float cy, const float cz) {
	
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id >= nParticles)
		return;

	float _r = r[id];
	float _vx = vx[id], _vy = vy[id], _vz = vz[id];
	float _v = _vx*_vx + _vy*_vy + _vz*_vz;
	float _v2 = _v*_v;

	float S = 4*3.14*_r*_r;

	//F = cx * 1/2 rho v^2 S
	fx[id] -= _vx* 1.0f/2.0f * cx * rho * _v2 * S;
	fy[id] -= _vy* 1.0f/2.0f * cy * rho * _v2 * S;
	fz[id] -= _vz* 1.0f/2.0f * cz * rho * _v2 * S;

}

__global__ void attractors(
		float *x, float *y, float *z, 
		float *fx, float *fy, float *fz,
		float *m, 
		const int nParticles, 
		const float dMin, const float dMax, 
		const float C) {
	
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id >= nParticles)
		return;

	float _x = x[id], _y = y[id], _z = z[id];
	float _m1 = m[id];

	float dx, dy, dz, d, d2;
	float _fx=0, _fy=0, _fz=0;
	float _C;

	for(int i = 0; i < nParticles; i++) {
		if(i==id)
			continue;

		dx = x[i] - _x;
		dy = y[i] - _y;
		dz = z[i] - _z;

		d2 = dx*dx + dy*dy + dz*dz;
		d = sqrt(d2);	

		if(d < dMin || d > dMax)
			continue;

		_C = C*_m1*m[i]/d2;
		
		_fx += _C * dx/d;
		_fy += _C * dy/d;
		_fz += _C * dz/d;
	}

	fx[id] += _fx;
	fy[id] += _fy;
	fz[id] += _fz;
}

	__global__ void
	__launch_bounds__(512)
	computeSprings(
				unsigned int *id1, unsigned int *id2,
				float *x, float *y, float *z,
				float *vx, float *vy, float *vz,
				float *fx, float *fy, float *fz,
				float *k, float *Lo, float *d, float *Fmax,
				bool *kill, float *intensity, float *outputLines,
				const bool handleDumping, 
				const unsigned int nSprings) {

	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id >= nSprings)
		return;
		
	if(kill[id])
		return;

	unsigned int _id1 = id1[id];
	unsigned int _id2 = id2[id];


	float _x1 = x[_id1], _y1 = y[_id1], _z1 = z[_id1];
	float _x2 = x[_id2], _y2 = y[_id2], _z2 = z[_id2];
	
	/*printf("\n(%f,%f,%f) \t(%f,%f,%f)", _x1, _y1, _z1, _x2, _y2, _z2 );*/
	
	float dx = _x2 - _x1; 
	float dy = _y2 - _y1; 
	float dz = _z2 - _z1;

	/*printf("\n%f %f %f", dx, dy, dz);*/
	
	float N = sqrt(dx*dx + dy*dy + dz*dz);

	if(N<1.0e-4)
		return; //null force

	float _k = k[id];
	float _Lo = Lo[id];
	float _d = d[id];
	float _Fmax = Fmax[id];
	
	/*printf("\n%i %f %f %f %f", id, _k, _Lo, _d, _Fmax);*/

	float dF, dFs, dFd, dfx, dfy, dfz;

	//stiffness
	dFs = - _k*(N - _Lo);
	/*printf("\ndFs %f", dFs);*/

	//damping
	dFd = 0;
	if(handleDumping && _d>1.0e-6) {
		float _vx1 = vx[_id1], _vy1 = vy[_id1], _vz1 = vz[_id1];
		float _vx2 = vx[_id2], _vy2 = vy[_id2], _vz2 = vz[_id2];
		
		float dvx = _vx2 - _vx1;
		float dvy = _vy2 - _vy1;
		float dvz = _vz2 - _vz1;
		dFd = -_d*(dvx*dx + dvy*dy + dvz*dz)/N;
	}
	
	//total force
	dF = dFs + dFd;
	/*if(dF > _Fmax)*/
		/*kill[id] = true;*/

	dF /= N;
	dfx = dF*dx;
	dfy = dF*dy;
	dfz = dF*dz;

	//update force on particles
	fx[_id1] -= dfx;
	fy[_id1] -= dfy;
	fz[_id1] -= dfz;
	
	fx[_id2] += dfx;
	fy[_id2] += dfy;
	fz[_id2] += dfz;

	//update vbos
	outputLines[6*id+0] = _x1;
	outputLines[6*id+1] = _y1;
	outputLines[6*id+2] = _z1;
	
	outputLines[6*id+3] = _x2;
	outputLines[6*id+4] = _y2;
	outputLines[6*id+5] = _z2;

	intensity[2*id+0] = min(abs(dF)/_Fmax,1.0f);
	intensity[2*id+1] = min(abs(dF)/_Fmax,1.0f);

	/*printf("\n%f", intensity[id]);*/
	/*printf("\n(%f,%f,%f) \t(%f,%f,%f)", _x1, _y1, _z1, _x2, _y2, _z2 );*/
}

				
	
__global__ void dynamicScheme(
		float *x, float *y, float *z,
		float *vx, float *vy, float *vz,
		float *fx, float *fy, float *fz,
		float *im,
		float dt,
		unsigned int nParticles) {

	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id >= nParticles)
		return;

	float inverseMass = im[id];
	
	vx[id] += dt*fx[id]*inverseMass;
	vy[id] += dt*fy[id]*inverseMass;
	vz[id] += dt*fz[id]*inverseMass;
	
	x[id] += vx[id]*dt;
	y[id] += vy[id]*dt;
	z[id] += vz[id]*dt;
	
	fx[id] = 0;
	fy[id] = 0;	
	fz[id] = 0;
}

void forceConstanteKernel(
		const struct mappedParticlePointers *pt, const unsigned int nParticles, 
		const float Fx, const float Fy, const float Fz) {
	
	dim3 blockDim(512,1,1);
	dim3 gridDim(ceil((float)nParticles/512),1,1);

	forceConstante<<<gridDim,blockDim,0,0>>>(
		pt->fx, pt->fy, pt->fz,
		nParticles,
		Fx, Fy, Fz);

	cudaDeviceSynchronize();
	checkKernelExecution();
}

void forceMassiqueConstanteKernel(
		const struct mappedParticlePointers *pt, const unsigned int nParticles, 
		const float mFx, const float mFy, const float mFz) {
	
	dim3 blockDim(512,1,1);
	dim3 gridDim(ceil((float)nParticles/512),1,1);

	forceMassiqueConstante<<<gridDim,blockDim,0,0>>>(
		pt->fx, pt->fy, pt->fz,
		pt->m, 
		nParticles,
		mFx, mFy, mFz);

	cudaDeviceSynchronize();
	checkKernelExecution();
}

void pousseeArchimedeKernel(
		const struct mappedParticlePointers *pt, const unsigned int nParticles, 
		const float nx, const float ny, const float nz, 
		const float rho, const float g) {
	
	dim3 blockDim(512,1,1);
	dim3 gridDim(ceil((float)nParticles/512),1,1);

	pousseeArchimede<<<gridDim,blockDim,0,0>>>(
		pt->x, pt->y, pt->z, 
		pt->fx, pt->fy, pt->fz,
		pt->r,  
		nParticles,
		nx, ny, nz,
		rho, g);

	cudaDeviceSynchronize();
	checkKernelExecution();
}

void frottementFluideKernel(
		const struct mappedParticlePointers *pt, const unsigned int nParticles, 
		const float k1, const float k2) {

	dim3 blockDim(512,1,1);
	dim3 gridDim(ceil((float)nParticles/512),1,1);

	frottementFluide<<<gridDim,blockDim,0,0>>>(
		pt->x, pt->y, pt->z, 
		pt->vx, pt->vy, pt->vz,
		pt->fx, pt->fy, pt->fz,
		nParticles, 
		k1, k2);
	
	cudaDeviceSynchronize();
	checkKernelExecution();
}

void frottementFluideAvanceKernel(
		const struct mappedParticlePointers *pt, const unsigned int nParticles, 
		const float rho, 
		const float cx, const float cy, const float cz) {

	dim3 blockDim(512,1,1);
	dim3 gridDim(ceil((float)nParticles/512),1,1);

	frottementFluideAvance<<<gridDim,blockDim,0,0>>>(
		pt->x, pt->y, pt->z, 
		pt->vx, pt->vy, pt->vz,
		pt->fx, pt->fy, pt->fz,
		pt->r,
		nParticles, 
		rho,
		cx, cy, cz);
	
	cudaDeviceSynchronize();
	checkKernelExecution();
}

void dynamicSchemeKernel(const struct mappedParticlePointers *pt, unsigned int nParticles) { 
	dim3 blockDim(512,1,1);
	dim3 gridDim(ceil((float)nParticles/512),1,1);

	float dt = 0.01;

	dynamicScheme<<<gridDim,blockDim,0,0>>>(
		pt->x, pt->y, pt->z, 
		pt->vx, pt->vy, pt->vz,
		pt->fx, pt->fy, pt->fz,
		pt->im, dt, nParticles);

	cudaDeviceSynchronize();
	checkKernelExecution();
}

void attractorKernel(const struct mappedParticlePointers *pt, 
		const unsigned int nParticles,
		const float dMin, const float dMax, const float C) { 

	dim3 blockDim(512,1,1);
	dim3 gridDim(ceil((float)nParticles/512),1,1);

	attractors<<<gridDim,blockDim,0,0>>>(
		pt->x, pt->y, pt->z, 
		pt->fx, pt->fy, pt->fz,
		pt->m, 
		nParticles,
		dMin, dMax, C);

	cudaDeviceSynchronize();
	checkKernelExecution();
}

void springKernel(const struct mappedParticlePointers *pt,
		const unsigned int nSprings,
		const bool handleDumping) {
	
	dim3 blockDim(512,1,1);
	dim3 gridDim(ceil((float)nSprings/512),1,1);

	computeSprings<<<gridDim,blockDim,0,0>>>(
				pt->id1, pt->id2,
				pt->x, pt->y, pt->z,
				pt->vx, pt->vy, pt->vz,
				pt->fx, pt->fy, pt->fz,
				pt->k, pt->Lo, pt->d, pt->Fmax,
				pt->killSpring, pt->intensity, pt->lines,
				handleDumping, 
				nSprings);
	
	cudaDeviceSynchronize();
	checkKernelExecution();
}
