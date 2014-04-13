
#ifndef KERNEL_H
#define KERNEL_H

#include "particleGroup.h"

extern void moveVertexKernel(float *vertex, unsigned int nVertex, float dx);
extern void moveKernel(const struct mappedParticlePointers* pt, unsigned int nParticles);

#endif /* end of include guard: KERNEL_H */
