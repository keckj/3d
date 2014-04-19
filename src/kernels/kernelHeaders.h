#ifndef KERNELHEADERS_H
#define KERNELHEADERS_H

#include "particleGroup.h"

extern void moveVertexKernel(float *vertex, unsigned int nVertex, float dx);
extern void moveKernel(const struct mappedParticlePointers* pt, unsigned int nParticles);

#endif /* end of include guard: KERNELHEADERS_H */
