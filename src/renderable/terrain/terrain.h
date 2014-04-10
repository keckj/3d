
#ifndef TERRAIN_H
#define TERRAIN_H

#include "consts.h"
#include "program.h"
#include "renderable.h"
#include "texture.h"

class Terrain : public Renderable {

	public:
		Terrain(unsigned char *heightmap, unsigned int width, unsigned int height, bool centered);
		~Terrain();
		
		void draw();
		
		const float *getRelativeModelMatrix() const;
	
	private:
		Program *program;
		Texture **textures;
		unsigned int VAO, VBO;
		std::map<std::string,int> uniformLocs;

		unsigned int width, height;
		bool centered;
		
		unsigned int nVertex;
		float *vertex, *colors;

		void writeColor(int height, unsigned int &idx, float *color);
		void writeVec3f(float *array, unsigned int &idx, float x, float y, float z);
	
		void inline makeProgram();
		void inline sendToDevice();
};

#endif /* end of include guard: TERRAIN_H */
