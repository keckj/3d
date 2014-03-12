#pragma once

#include "renderable.hpp"
#include <GL/glew.h>

class Box : public Renderable {
	
	public:
		Box(float width, float height, float length, 
		float x=0.0f, float y=0.0f, float z=0.0f, bool centered = true,
		unsigned int program=0, int modelMatrixLocation=0); 
		
		~Box();

		void setPosition(float x, float y, float z);
		
		void draw(const float *modelMatrix = consts::identity) const;

		const float *getRelativeModelMatrix() const;

	private:	
		float width, height, length;
		float x, y, z;
		bool centered;

		unsigned int program;
		int modelMatrixLocation;

		unsigned int VAO, VBO;
		
		float *vertex, *normals, *colors;

		void inline computeVertex();
		void inline sendToDevice();

		void writeVec3f(float *array, unsigned int &idx, float x, float y, float z);

};
