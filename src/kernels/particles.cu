

#include "cuda.h"
#include "cuda_runtime.h"
#include "stdio.h"

#include <iostream>

struct mappedParticlePointers {
	float *x, *y, *z, *vx, *vy, *vz, *m, *im, *r, *kill;
};

extern void checkKernelExecution();

__global__ void move(float *x, float *y, float *z, 
		float *vx, float *vy, float *vz,
		float *r, float *m, float *im, 
		int nParticles) {

	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id > nParticles)
		return;
	/*if(id == 0)*/
		/*printf("%f %f %f\n%f %f %f\n%f %f %f\n %i\n\n",x[id],y[id],z[id],vx[id],vy[id],vz[id],r[id],m[id],im[id],nParticles);*/

	float _r = r[id];
	float _m = m[id];
	float _im = im[id];

	float fx = 0.0f, fy = 0.0f, fz = 0.0f;
	float dt = 0.01f;
	
	float S = 4*3.14*_r*_r; 
	float V = 4.0/3.0*3.14*_r*_r*_r;
	
	float rhoEau = 1000;
	float g = -9.81;
	
	fy+= 5*sqrt(r[id]);
	fy-= vy[id]*S;

	/*fy -= rhoEau*g*V + _m*g;*/

	//float dx, dy, dz, d, d2, n;
	/*for(int i = 0; i < nParticles; i++) {*/
		/*if(i==id)*/
			/*continue;*/

		/*dx = x[i] - x[id];*/
		/*dy = y[i] - y[id];*/
		/*dz = z[i] - z[id];*/
		/*d2 = dx*dx + dy*dy + dz*dz;*/
		/*d = sqrt(d2);*/

		/*n = G*m[i]*m[id]/d2;*/
		/*fx -= n*dx/d;*/
		/*fy -= n*dy/d;*/
		/*fz -= n*dz/d;*/
	/*}*/

	/*fx-= nu*vx[id]*S;	*/
	/*fy-= nu*vy[id]*S;	*/
	/*fz-= nu*vz[id]*S;	*/

	
	vx[id] += dt*fx*_im;
	vy[id] += dt*fy*_im;
	vz[id] += dt*fz*_im;

	x[id] += vx[id]*dt;
	y[id] += vy[id]*dt;
	z[id] += vz[id]*dt;
	
}

void moveKernel(const struct mappedParticlePointers *pt, unsigned int nParticles) {
	
	dim3 blockDim(512,1,1);
	dim3 gridDim(ceil((float)nParticles/512),1,1);

	move<<<gridDim,blockDim,0,0>>>(pt->x, pt->y, pt->z, 
		pt->vx, pt->vy, pt->vz,
		pt->r, pt->m, pt->im, 
		nParticles);
	cudaDeviceSynchronize();
	checkKernelExecution();
}

