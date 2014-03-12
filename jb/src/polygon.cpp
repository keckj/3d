
#include "polygon.hpp"

#include <iostream>
#include <cstdio>

using namespace std;

Polygon::Polygon(float *points, float *colours, size_t size) {
	
	Polygon::points = points;
	Polygon::size = size;
	
	glGenBuffers(1, &vertexBufferObject);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
	glBufferData(GL_ARRAY_BUFFER, size * sizeof(float), points, GL_STATIC_DRAW);
	
	glGenBuffers (1, &colorBufferObject);
	glBindBuffer (GL_ARRAY_BUFFER, colorBufferObject);
	glBufferData (GL_ARRAY_BUFFER, size * sizeof (float), colours, GL_STATIC_DRAW);
}

float *Polygon::getPoints() {
	return points;
}

size_t Polygon::getSize() {
	return size;
}

unsigned int Polygon::getVertexBufferObject() {
	return vertexBufferObject;
}

unsigned int Polygon::getColorBufferObject() {
	return colorBufferObject;
}
