
#ifndef MARCHINGCUBES_H
#define MARCHINGCUBES_H

#include "renderTree.h"
#include "program.h"
#include "consts.h"
#include "texture3D.h"

class MarchingCubes : public RenderTree {

	public:
		MarchingCubes(unsigned int width=256, unsigned int height=256, unsigned int length=4);	
		~MarchingCubes();	

	private:
		Texture *_density;
		unsigned int _width, _height, _length;

		Program *_drawProgram, *_densityProgram;
		std::map<std::string,int> _drawUniformLocs, _densityUniformLocs;

		unsigned int vertexVBO, fullscreenQuadVBO;
		
		void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
		
		void makeDrawProgram();
		void makeDensityProgram();

		void generateQuads();
		void generateFullScreenQuad();
};


#endif /* end of include guard: MARCHINGCUBES_H */
