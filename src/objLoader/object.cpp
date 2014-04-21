#include <cstring>

#include "log.h"
#include "terrain.h"
#include "matrix.h"
#include <GL/glew.h>

#include "object.h"

Object::Object(tinyobj::shape_t shape, Program *program) : 
    shape(shape), program(program), textures(0), VAO(0), VBO(0)
{
    initializeRelativeModelMatrix();

    makeTextures();
   
    sendToDevice(); 
}

Object::~Object() {
    /*for (unsigned int i = 0; i < nTextures; i++) {
        delete textures[i];   
    }

    delete[] textures;*/
}

//only JPG textures supported
void inline Object::makeTextures() {
/*
    // We only use map_Ka and map_Kd in the shader
    nTextures = 0;
    Texture2D *tex_a = NULL;
    Texture2D *tex_d = NULL;

    if (shape.material.ambient_texname.c_str()) {
        nTextures++;
        tex_a = new Texture2D(shape.material.ambient_texname.c_str() ,"jpg");
        tex_a->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_REPEAT));
		tex_a->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_REPEAT));
		tex_a->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
		tex_a->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		tex_a->generateMipMap();
    }
    if (shape.material.diffuse_texname.c_str()) {
        nTextures++;
        tex_b = new Texture2D(shape.material.diffuse_texname.c_str(), "jpg");
        tex_b->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_REPEAT));
		tex_b->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_REPEAT));
		tex_b->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
		tex_b->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		tex_b->generateMipMap();
    }

    if (nTextures > 0) {
        textures = new Texture*[nTextures];
        textures[0] = tex_a;
        textures[1] = tex_b;
    }

    //TODO signaler au shader qu'on utilise que certaines ou pas de textures*/
}

void inline Object::sendToDevice() {
  
    // 4 VBOs : positions, normals, texCoords, indices
    VBO = new GLuint[3];
    glGenBuffers(3, VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, shape.mesh.positions.size()*sizeof(GLfloat), &shape.mesh.positions[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, shape.mesh.normals.size()*sizeof(GLfloat), &shape.mesh.normals[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
    glBufferData(GL_ARRAY_BUFFER, shape.mesh.texcoords.size()*sizeof(GLfloat), &shape.mesh.texcoords[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO[3]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, shape.mesh.indices.size()*sizeof(GLuint), &shape.mesh.indices[0], GL_STATIC_DRAW);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    {
        glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), NULL);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), NULL);
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), NULL);
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO[3]);
    }
    glBindVertexArray(0);
}

void Object::drawDownwards(const float *currentTransformationMatrix) {
	/*static float *proj = new float[16], *view = new float[16];

	program->use();

	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glUniformMatrix4fv(uniformLocs["projectionMatrix"], 1, GL_FALSE, proj);
	glUniformMatrix4fv(uniformLocs["viewMatrix"], 1, GL_FALSE, view);
	glUniformMatrix4fv(uniformLocs["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);*/

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, shape.mesh.indices.size(), GL_UNSIGNED_INT, 0);
}

void Object::initializeRelativeModelMatrix() {
   
    // Initialize the object at (0,0) and with default rotation
    const float mat[] = {
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f
    };

    setRelativeModelMatrix(mat);
}
