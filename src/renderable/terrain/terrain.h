
#pragma once

#include "consts.hpp"
#include "renderable.h"

class Terrain : public Renderable {

	public:
		Terrain(unsigned char *heightmap, unsigned int width, unsigned int height, bool centered, 
			unsigned int program, 
			unsigned int modelMatrixLocation, unsigned int projectionMatrixLocation, unsigned int viewMatrixLocation);
		~Terrain();
		
		void draw();
		
		const float *getRelativeModelMatrix() const;

	
	private:
		unsigned int width, height;
		bool centered;
		unsigned int program, modelMatrixLocation, projectionMatrixLocation, viewMatrixLocation;
		unsigned int VAO, VBO;
		
		unsigned int nVertex;
		float *vertex, *colors;
		void writeColor(int height, unsigned int &idx, float *color);
		void writeVec3f(float *array, unsigned int &idx, float x, float y, float z);
	
		void inline sendToDevice();
};
