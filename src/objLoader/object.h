#ifndef OBJECT_H
#define OBJECT_H

#include <map>
#include <cstring>

#include "tiny_obj_loader.h"

#include "consts.h"
#include "program.h"
#include "renderTree.h"
#include "texture2D.h"

typedef struct Light {
    GLfloat position[4];
    GLfloat diffuse[4];
    GLfloat specular[4];
    GLfloat spotDirection[4]; // w useless, used for padding for std140
    GLfloat constantAttenuation, linearAttenuation, quadraticAttenuation;
    GLfloat spotCutoff, spotExponent;
    GLfloat isEnabled; // bool
    GLfloat dum[2]; // struct padding to 6*vec4
} Light_s;

typedef struct Material {
    GLfloat ambient[4];
    GLfloat diffuse[4];
    GLfloat specular[4];
    GLfloat shininess, transparency;
    GLfloat hasTexture; // bool
    GLfloat dummy; // struct padding to 4*vec4
} Material_s;


class Object : public RenderTree {

	public:
		Object(tinyobj::shape_t shape);
		~Object();
		
	private:
        tinyobj::shape_t shape;

		Program *program;
		Texture **textures;
        unsigned int nTextures;
		std::map<std::string,int> uniformLocs;
		GLuint VAO;
        GLuint *VBO;
        GLuint lightsUBO, materialUBO;

        void inline createUBOs();
		void inline makeProgram();
		void inline sendToDevice();
		
		void drawDownwards(const float *currentTransformationMatrix = consts::identity4);

		void initializeRelativeModelMatrix();
};

#endif /* end of include guard: OBJECT_H */
