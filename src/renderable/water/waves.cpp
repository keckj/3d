#include <GL/glew.h>
#include <cmath>
#include <iostream> 
#include <cstdio>
#include <ctime>
#include <cassert>

#include "program.h"
#include "globals.h"
#include "shader.h"
#include "viewer.h"
#include "audible.h"
#include "waves.h"
#include "consts.h"
#include "matrix.h"

using namespace Matrix;

#define N_MOBILES_X 256
#define N_MOBILES_Z 256

Waves::~Waves() {
    delete[] mobiles;
    delete[] indices;
}

// The drawn square will be centered on (xPos,zPos).
Waves::Waves(float xPos, float zPos, float xWidth, float zWidth, float meanHeight, Texture *cubeMapTexture) :
program("Waves") { 

    if (xWidth <= 0.0f || zWidth <= 0.0f || meanHeight <= 0.0f) {
        std::cout << "You're doing something stupid!" << std::endl;
        return;
    }

    this->xPos = xPos;
    this->zPos = zPos;
    this->xWidth = xWidth;
    this->zWidth = zWidth;
    this->meanHeight = meanHeight;
    this->deltaX = 1.0/(N_MOBILES_X-1);
    this->deltaZ = 1.0/(N_MOBILES_Z-1);

    this->stopAnimating = false;

	initializeRelativeModelMatrix();

    // Ambient sounds
    this->underwaterSound = new Audible("sounds/ambiant/Underwater_Pool_converted.wav", qglviewer::Vec(0,0,0));
    this->underwaterSoundPlaying = false;

    // Indices used for drawing
    nIndices = 6*(N_MOBILES_X-1)*(N_MOBILES_Z-1);
    indices = new GLuint[nIndices];
    int currentIndex = 0;

    nMobiles = N_MOBILES_X * N_MOBILES_Z;
    mobiles = new Mobile[nMobiles];
    for (int x = 0; x < N_MOBILES_X; x++) {
        for (int z = 0; z < N_MOBILES_Z; z++) {
            int idx = x*N_MOBILES_Z + z;
            mobiles[idx].x = 1.0/(N_MOBILES_X-1) * (x - (N_MOBILES_X-1)/2);  
            mobiles[idx].y = 0.0f;
            mobiles[idx].z = 1.0/(N_MOBILES_Z-1) * (z - (N_MOBILES_Z-1)/2); 

            // Construct triangles
            if (x < N_MOBILES_X-1 && z < N_MOBILES_Z-1) { 
                // 1st triangle
                indices[currentIndex++] = idx;                  // up left
                indices[currentIndex++] = idx+1;                // up right
                indices[currentIndex++] = idx+1+N_MOBILES_Z;    // down right
                // 2nd triangle
                indices[currentIndex++] = idx;                  // up left
                indices[currentIndex++] = idx+1+N_MOBILES_Z;    // down right
                indices[currentIndex++] = idx+N_MOBILES_Z;      // down left
            }
        }
    }

    time = 0.0f;

    // -- attribs --
	program.bindAttribLocation(0, "position");
	program.bindFragDataLocation(0, "out_color");
	
    // -- shaders --
	program.attachShader(Shader("shaders/waves/water.vert", GL_VERTEX_SHADER));
	program.attachShader(Shader("shaders/waves/water.frag", GL_FRAGMENT_SHADER));

    // -- linkage --
	program.link();

    // -- uniforms --
	uniformLocs = program.getUniformLocationsMap("modelMatrix viewMatrix projectionMatrix invView time deltaX deltaZ", false);

    // -- cube map --
    if (cubeMapTexture == NULL) {
        log_console.errorStream() << "[WAVES.CPP] cubeMapTexture == NULL !";
        exit(1);
    }
    Texture* textures[1];
    textures[0] = cubeMapTexture;
    program.bindTextures(textures, "cubeMapTexture", "true");

    // -- VBOs --
    vertexBuffers = new GLuint[2];
    glGenBuffers(2, vertexBuffers);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffers[0]);
    glBufferData(GL_ARRAY_BUFFER, nMobiles*sizeof(Mobile), mobiles, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBuffers[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, nIndices*sizeof(GLuint), indices, GL_STATIC_DRAW);

    // -- VAO --
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);
    {   
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffers[0]);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); // position
        glEnableVertexAttribArray(0); // 0 <=> position

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBuffers[1]);
    }
    glBindVertexArray(0);
}


void Waves::drawDownwards(const float *currentTransformationMatrix) {
        
    float *proj = new float[16], *view = new float[16];

	program.use();

	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
    float *viewInv = inverseMat4f(view);
    //std::cout << viewInv[0] << "/" << viewInv[1] << "/" << viewInv[2] << "/" << viewInv[3] << "\n" << viewInv[4] << "/" << viewInv[5] << "/" << viewInv[6] << "/" << viewInv[7] << "\n" << viewInv[8] << "/" << viewInv[9] << "/" << viewInv[10] << "/" << viewInv[11] << "\n" << viewInv[12] << "/" << viewInv[13] << "/" << viewInv[14] << "/" << viewInv[15] << "\n\n"; 

	glUniformMatrix4fv(uniformLocs["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);
	glUniformMatrix4fv(uniformLocs["viewMatrix"], 1, GL_FALSE, view);
	glUniformMatrix4fv(uniformLocs["projectionMatrix"], 1, GL_FALSE, proj);
    glUniformMatrix4fv(uniformLocs["invView"], 1, GL_FALSE, viewInv);
    glUniform1f(uniformLocs["time"], time);
    glUniform1f(uniformLocs["deltaX"], deltaX);
    glUniform1f(uniformLocs["deltaZ"], deltaZ);

    glBindVertexArray(vertexArray);    
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glDrawElements(GL_TRIANGLES, nIndices, GL_UNSIGNED_INT, 0);

    glDisable(GL_BLEND);

    glBindVertexArray(0);
	glUseProgram(0);
}


void Waves::animateDownwards() {

    // Check if we were told to stop animating
    if (stopAnimating) return;

    float deltaT = 1.0 / 50;
    time += deltaT;

    //std::cout << "time=" << time << std::endl;
    
    // Ambient sounds
    if (Globals::viewer->camera()->position()[1] < meanHeight) {
        if (!underwaterSoundPlaying) {
            underwaterSound->setGain(max(Globals::viewer->camera()->position().squaredNorm(), 1.0f));
            underwaterSound->playSource();
            underwaterSoundPlaying = true;
        }
    } else {
        if (underwaterSoundPlaying) {
            underwaterSound->pauseSource();
            underwaterSoundPlaying = false;
        }
    }
}


void Waves::initializeRelativeModelMatrix() {

	const float m[] = {
		xWidth, 0.0f, 0.0f, xPos,
		0.0f, 1.0f, 0.0f, meanHeight,
		0.0f, 0.0f, zWidth, zPos,
		0.0f, 0.0f, 0.0f, 1.0f
	};
	
	setRelativeModelMatrix(m);
}


void Waves::keyPressEvent(QKeyEvent *e) {
    if (e->key() == Qt::Key_P && e->modifiers() == Qt::NoButton) {
        stopAnimating = !stopAnimating;
    }
}
