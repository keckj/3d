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
#include "waves.h"


static const int N_MOBILES_X = 256;
static const int N_MOBILES_Z = 256;

Waves::~Waves() {
    delete[] mobiles;
    delete[] indices;
}

// The drawn square will be centered on (xPos,zPos).
Waves::Waves(float xPos, float zPos, float xWidth, float zWidth, float meanHeight) :
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
	program.attachShader(Shader("shaders/water.vert", GL_VERTEX_SHADER));
	program.attachShader(Shader("shaders/water.frag", GL_FRAGMENT_SHADER));

    // -- linkage --
	program.link();

    // -- uniforms --
	uniforms_vec = program.getUniformLocations("modelMatrix viewMatrix projectionMatrix time deltaX deltaZ", true);

	program.use();
    
    // -- VBOs --
    vertexBuffers = new GLuint[2];
    glGenBuffers(2, vertexBuffers);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffers[0]);
    glBufferData(GL_ARRAY_BUFFER, nMobiles*sizeof(Mobile), mobiles, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBuffers[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, nIndices*sizeof(GLuint), indices, GL_STATIC_DRAW);
}


void Waves::draw() {
        
    static float *proj = new float[16], *view = new float[16];

	program.use();
	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glUniformMatrix4fv(uniforms_vec[0], 1, GL_TRUE, getModelMatrix());
	glUniformMatrix4fv(uniforms_vec[1], 1, GL_FALSE, view);
	glUniformMatrix4fv(uniforms_vec[2], 1, GL_FALSE, proj);
    glUniform1f(uniforms_vec[3], time);
    glUniform1f(uniforms_vec[4], deltaX);
    glUniform1f(uniforms_vec[5], deltaZ);

    /*//Enable the vertex array functionality
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(	3,
            GL_FLOAT,
            sizeof(Mobile),
            mobiles);

    glDrawElements(GL_TRIANGLES, nIndices, GL_UNSIGNED_INT, indices);
    glDisableClientState(GL_VERTEX_ARRAY);*/

    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffers[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0); // position
    glEnableVertexAttribArray(0); // 0 <=> position

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBuffers[1]);

    glDrawElements(GL_TRIANGLES, nIndices, GL_UNSIGNED_INT, 0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
    //glDisableVertexAttribArray(0); // position

	glUseProgram(0);
}


void Waves::animate() {

    // Check if we were told to stop animating
    if (stopAnimating) return;

    float deltaT = 1.0 / 50;
    time += deltaT;

    //std::cout << "time=" << time << std::endl;
}


float *Waves::getModelMatrix() const {

	const float m[] = {
		xWidth, 0.0f, 0.0f, xPos,
		0.0f, 1.0f, 0.0f, meanHeight,
		0.0f, 0.0f, zWidth, zPos,
		0.0f, 0.0f, 0.0f, 1.0f
	};
	
    float *mCpy = new float[16];
	memcpy(mCpy, m, 16*sizeof(float));

	return mCpy;
}


void Waves::keyPressEvent(QKeyEvent *e, Viewer &v) {
    if (e->key() == Qt::Key_P && e->modifiers() == Qt::NoButton) {
        stopAnimating = !stopAnimating;
    }
    (void)v; // suppress warning
}
