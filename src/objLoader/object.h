#ifndef OBJECT_H
#define OBJECT_H

#include <map>
#include <cstring>

#include "tiny_obj_loader.h"

#include "consts.h"
#include "program.h"
#include "renderTree.h"
#include "texture2D.h"

class Object : public RenderTree {

	public:
		Object(tinyobj::shape_t shape, Program *program);
		~Object();
		
	private:
        tinyobj::shape_t shape;

		Program *program;
		Texture **textures;
        unsigned int nTextures;
		GLuint VAO;
        GLuint *VBO;
		std::map<std::string,int> uniformLocs;

		void inline makeTextures();
		void inline sendToDevice();
		
		void drawDownwards(const float *currentTransformationMatrix = consts::identity4);

		void initializeRelativeModelMatrix();
};

#endif /* end of include guard: OBJECT_H */
