
#pragma once

#include "consts.hpp"
#include "renderable.hpp"

class Terrain : public Renderable {

	public:
		Terrain(unsigned char *heightmap, unsigned int width, unsigned int height, bool centered, 
				unsigned int modelMatrixLocation, unsigned int program);
		~Terrain();
		
		void draw(const float *modelMatrix = consts::identity) const;
		const float *getRelativeModelMatrix() const;

	
	private:
		unsigned int width, height;
		bool centered;
		unsigned int modelMatrixLocation, program;
		unsigned int VAO, VBO;
		
		unsigned int nVertex;
		float *vertex, *colors;
		void writeColor(int height, unsigned int &idx, float *color);
		void writeVec3f(float *array, unsigned int &idx, float x, float y, float z);
	
		void inline sendToDevice();
};
