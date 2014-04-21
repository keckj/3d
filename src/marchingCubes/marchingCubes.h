
#ifndef MARCHINGCUBES_H
#define MARCHINGCUBES_H

#include "renderTree.h"
#include "program.h"
#include "consts.h"
#include "texture3D.h"

//Les struct à envoyer en uniform
namespace MarchingCube {

	//un vec4 = {x,y,z,w}
	//Avec le format std140, les vec3 sont alignés sur 4 octets
	//(comme les vec4)
	//=> on rajoute des 0 pour faire des memcpy simplement

	//loockup tables pour l'algo MC
	typedef struct LookupTable {
		GLuint caseToNumPoly[1024]; //nb de triangle par cas (256 uint) y z w inutiles (alignement du tableau sur des vec4)
		GLfloat edgeStart[48];	   //debut des cotés (12 vec4) w inutile
		GLfloat edgeDir[48];	   //direction des cotés (12 vec4) w inutile
		GLfloat maskA0123[48];	   //masques pour retrouver la densité en fonction de l'arrete
		GLfloat maskB0123[48];	   //4*12 vec4
		GLfloat maskA4567[48];
		GLfloat maskB4567[48];
	} LookupTable_s;

	//table de triangles
	//max 5 triangle par cas, 3 cotés par triangle
	//environ 20.5kB attention au GL_MAX_UNIFORM_BLOCKSIZE 
	typedef struct TriTable {
		GLint triangleTable[5120]; //5*256 GLint4 w inutile
	} TriTable_s;

	//other data
	typedef struct GeneralData {
		GLfloat textureSize[4];   //vec4 w inutile
		GLfloat voxelGridSize[4]; //vec4 w inutile
		GLfloat voxelDim[4];      //vec4 w inutile
	} GeneralData_s;
}

class MarchingCubes : public RenderTree {

	public:
		MarchingCubes(unsigned int width=256, unsigned int height=256, unsigned int length=256, float voxelSize = 0.1f);	
		~MarchingCubes();	

	private:
		Texture *_density;
		unsigned int _textureWidth, _textureHeight, _textureLength;
		unsigned int _voxelGridWidth, _voxelGridHeight, _voxelGridLength;
		float _voxelWidth, _voxelHeight, _voxelLength;

		Program *_drawProgram, *_densityProgram, *_marchingCubesProgram;
		std::map<std::string,int> _drawUniformLocs, _densityUniformLocs, _marchingCubesUniformLocs; 
		std::map<std::string,int> _marchingCubesBlockIndexes;

		unsigned int _vertexVBO, _fullscreenQuadVBO, _marchingCubesLowerLeftXY_VBO;           

		unsigned int _generalDataUBO;

		void drawDownwards(const float *currentTransformationMatrix = consts::identity4);

		void makeDrawProgram();
		void makeDensityProgram();
		void makeMarchingCubesProgram();

		void generateQuads();
		void generateFullScreenQuad();
		void generateMarchingCubesPoints();

		static void generateUniformBlockBuffers();
		static unsigned int _triTableUBO, _lookupTableUBO;
};

#endif /* end of include guard: MARCHINGCUBES_H */
