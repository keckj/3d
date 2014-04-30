#ifndef __SKYBOX_H__
#define __SKYBOX_H__

#include "renderTree.h"
#include "cubeMap.h"
#include "program.h"
#include <string>

class Skybox : public RenderTree {
    public:
		//files order : POS_X NEG_X POS_Y NEG_Y POS_Z NEG_Z
        Skybox (const std::string &folder, const std::string &fileNames, const std::string &format);
        ~Skybox ();

		void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        Texture* getCubeMap();

    private:
        Texture *_cubeMap;
		Program *_program;

		std::map<std::string, int> _uniformLocations;

		void makeProgram();

		static void initVBOs();

		static float _vertexCoords[];
	
		static bool _init;
		static unsigned int _vertexVBO;
		static unsigned int _targetsVBO;
};
		

#endif

