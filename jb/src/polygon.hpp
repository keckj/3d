#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class Polygon {

	protected:
		float *points;
		size_t size;

		unsigned int vertexBufferObject;
		unsigned int colorBufferObject;
	public:
		Polygon(float *points, float *colors, size_t size);

		float *getPoints();
		size_t getSize();

		unsigned int getVertexBufferObject();
		unsigned int getColorBufferObject();
};
