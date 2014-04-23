#include <cstring>

#include "log.h"
#include "terrain.h"
#include "matrix.h"
#include <GL/glew.h>

#include "object.h"

Object::Object(tinyobj::shape_t shape, Program *program, std::map<std::string,int> uniformLocs) : 
    shape(shape), program(program), uniformLocs(uniformLocs), textures(0), VAO(0), VBO(0)
{
    initializeRelativeModelMatrix();

    createUBOs();

    makeTextures();
   
    sendToDevice(); 
}

Object::~Object() {
    /*for (unsigned int i = 0; i < nTextures; i++) {
        delete textures[i];   
    }

    delete[] textures;*/
}

void Object::createUBOs() {

    //test
    Light_s lightsData[5];
    float lightPos[] = {-1.0f, -1.0f, 0.0f, 0.0f};
    float lightDiffuse[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float lightSpecular[] = {0.0f, 1.0f, 0.0f, 1.0f};
    memcpy(lightsData[0].position, lightPos, 4*sizeof(GLfloat));
    memcpy(lightsData[0].diffuse, lightDiffuse, 4*sizeof(GLfloat));
    memcpy(lightsData[0].specular, lightSpecular, 4*sizeof(GLfloat));
    lightsData[0].constantAttenuation = 1.0f;
    lightsData[0].isEnabled = 1.0f;

    tinyobj::material_t mat = shape.material;
    Material_s materialData = {
                                mat.ambient[0],
                                mat.ambient[1],
                                mat.ambient[2],
                                0.0f,
                                mat.diffuse[0],
                                mat.diffuse[1],
                                mat.diffuse[2],
                                0.0f,
                                mat.specular[0],
                                mat.specular[1],
                                mat.specular[2],
                                0.0f,
                                mat.shininess,
                                mat.dissolve,
                                0.0f, // false for testing
                                0.0f
                               };

    glGenBuffers(1, &lightsUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, lightsUBO);
    glBufferData(GL_UNIFORM_BUFFER, 5*sizeof(Light_s), lightsData, GL_STATIC_DRAW);
    
    glGenBuffers(1, &materialUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, materialUBO);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(Material_s), &materialData, GL_STATIC_DRAW);
	
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

//only JPG textures supported for now
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
    VBO = new GLuint[4];
    glGenBuffers(4, VBO);
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
        if (shape.mesh.normals.size() > 0) {
            std::cout << "lol\n";
           glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
           glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), NULL);
           glEnableVertexAttribArray(1);
        }
        if (shape.mesh.texcoords.size() > 0) {
            std::cout << "lolt\n";
            glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), NULL);
            glEnableVertexAttribArray(2);
        }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO[3]);
    }
    glBindVertexArray(0);
}

void Object::drawDownwards(const float *currentTransformationMatrix) {
	static float *proj = new float[16], *view = new float[16];

	program->use();

	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glUniformMatrix4fv(uniformLocs["projectionMatrix"], 1, GL_FALSE, proj);
	glUniformMatrix4fv(uniformLocs["viewMatrix"], 1, GL_FALSE, view);
	glUniformMatrix4fv(uniformLocs["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);

    glBindBufferBase(GL_UNIFORM_BUFFER, 0, lightsUBO);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, materialUBO);

    glBindVertexArray(VAO);

    glDrawElements(GL_TRIANGLES, shape.mesh.indices.size(), GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);   
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, 0);
	glBindBufferBase(GL_UNIFORM_BUFFER, 1, 0);
	glUseProgram(0);
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
