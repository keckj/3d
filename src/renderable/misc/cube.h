#ifndef _CUBE_
#define _CUBE_

#include "renderable.h"
#include <GL/glut.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_gl_interop.h>


class Cube : public Renderable
{
	public:
		Cube();
		~Cube();
		void draw();
		void animate();

	private:
		unsigned int program,vao, vbo;
		cudaGraphicsResource *cudaVbo;
		int modelMatrixLocation, viewMatrixLocation, projectionMatrixLocation;

		unsigned int size;
		unsigned int subsize;

		unsigned int source;
};

#endif

