#include <iostream>
#include <GL/glew.h>
#include <cassert>

#include "cube.h"
#include "consts.h"
#include "shader.h"
#include "globals.h"
#include "kernel.h"
#include "matrix.h"
#include "cudaUtils.h"
#include "audible.h"

using namespace std;

static GLfloat allVertices[] = { 
	+0.5, -0.5, -0.5,   -0.5, -0.5, -0.5,   -0.5, +0.5, -0.5,   +0.5, +0.5, -0.5,  // 0 3 2 1
	+0.5, -0.5, -0.5,   +0.5, -0.5, +0.5,   -0.5, -0.5, +0.5,   -0.5, -0.5, -0.5,  // 0 4 7 3
	+0.5, +0.5, -0.5,   +0.5, +0.5, +0.5,   +0.5, -0.5, +0.5,   +0.5, -0.5, -0.5,  // 1 5 4 0
	-0.5, +0.5, -0.5,   -0.5, +0.5, +0.5,   +0.5, +0.5, +0.5,   +0.5, +0.5, -0.5,  // 2 6 5 1
	-0.5, -0.5, -0.5,   -0.5, -0.5, +0.5,   -0.5, +0.5, +0.5,   -0.5, +0.5, -0.5,  // 3 7 6 2
	+0.5, -0.5, +0.5,   +0.5, +0.5, +0.5,   -0.5, +0.5, +0.5,   -0.5, -0.5, +0.5,  // 4 5 6 7 
};

static GLfloat allNormals[] = { 
	0.0,  0.0, -1.0,    0.0,  0.0, -1.0,    0.0,  0.0, -1.0,    0.0,  0.0, -1.0,  // 0 3 2 1
	0.0, -1.0,  0.0,    0.0, -1.0,  0.0,    0.0, -1.0,  0.0,    0.0, -1.0,  0.0,  // 0 4 7 3
	1.0,  0.0,  0.0,    1.0,  0.0,  0.0,    1.0,  0.0,  0.0,    1.0,  0.0,  0.0,  // 1 5 4 0
	0.0,  1.0,  0.0,    0.0,  1.0,  0.0,    0.0,  1.0,  0.0,    0.0,  1.0,  0.0,  // 2 6 5 1
	-1.0,  0.0,  0.0,   -1.0,  0.0,  0.0,   -1.0,  0.0,  0.0,   -1.0,  0.0,  0.0,  // 3 7 6 2
	0.0,  0.0,  1.0,    0.0,  0.0,  1.0,    0.0,  0.0,  1.0,    0.0,  0.0,  1.0,  // 4 5 6 7
};

static GLfloat allColors[] = { 
	0.0,  0.0, 1.0,    0.0,  0.0, 1.0,    0.0,  0.0, 1.0,    0.0,  0.0, 1.0,  // 0 3 2 1
	0.0, 1.0,  0.0,    0.0, 1.0,  0.0,    0.0, 1.0,  0.0,    0.0, 1.0,  0.0,  // 0 4 7 3
	1.0,  0.0,  0.0,    1.0,  0.0,  0.0,    1.0,  0.0,  0.0,    1.0,  0.0,  0.0,  // 1 5 4 0
	0.0,  1.0,  0.0,    0.0,  1.0,  0.0,    0.0,  1.0,  0.0,    0.0,  1.0,  0.0,  // 2 6 5 1
	1.0,  0.0,  0.0,   1.0,  0.0,  0.0,   1.0,  0.0,  0.0,   1.0,  0.0,  0.0,  // 3 7 6 2
	0.0,  0.0,  1.0,    0.0,  0.0,  1.0,    0.0,  0.0,  1.0,    0.0,  0.0,  1.0,  // 4 5 6 7
};


Cube::Cube() {

	unsigned int nbSubBuffer = 3;
	subsize = 6*4*3;
	size = subsize*nbSubBuffer;

	//create vbo, allocate data
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size*sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0*subsize*sizeof(float), subsize*sizeof(float), allVertices);
	glBufferSubData(GL_ARRAY_BUFFER, 1*subsize*sizeof(float), subsize*sizeof(float), allColors);
	glBufferSubData(GL_ARRAY_BUFFER, 2*subsize*sizeof(float), subsize*sizeof(float), allNormals);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//register vbo to cuda (flagsnone = read/write)	
	cudaGraphicsGLRegisterBuffer(&cudaVbo, vbo, cudaGraphicsMapFlagsNone);

	//create vao
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	{	
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (char*)NULL + subsize*sizeof(float));
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (char*)NULL + 2*subsize*sizeof(float));
		glEnableVertexAttribArray(2);
	}
	glBindVertexArray(0);

	//create program
	program = glCreateProgram ();
	Shader vs("shaders/common/vs.glsl", GL_VERTEX_SHADER);
	Shader fs("shaders/common/fs.glsl", GL_FRAGMENT_SHADER);
	glAttachShader (program, fs.getShader());
	glAttachShader (program, vs.getShader());
	glBindAttribLocation(program, 0, "vertex_position");
	glBindAttribLocation(program, 1, "vertex_colour");
	glBindAttribLocation(program, 2, "vertex_normal");
	glBindFragDataLocation(program, 0, "out_colour");
	glLinkProgram(program);
	int status;
  	glGetProgramiv(program, GL_LINK_STATUS, &status);
	assert(status);

	glUseProgram(program);
	modelMatrixLocation = glGetUniformLocation(program, "modelMatrix");
	viewMatrixLocation = glGetUniformLocation(program, "viewMatrix");
	projectionMatrixLocation = glGetUniformLocation(program, "projectionMatrix");
	assert(modelMatrixLocation != -1);
	assert(viewMatrixLocation != -1);
	assert(projectionMatrixLocation != -1);
	glUseProgram(0);



	ALfloat listenerOri[] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f };
	alListener3f(AL_POSITION, 0, 0, 0);
	alListener3f(AL_VELOCITY, 0, 0, 0);
	alListenerfv(AL_ORIENTATION, listenerOri);

	Audible *test = new Audible("sounds/ambiant/waves_converted.wav", qglviewer::Vec(0,0,0));
	test->setGain(5.0f);
	test->playSource();
}

Cube::~Cube() {
	cudaGraphicsUnmapResources(1, &cudaVbo, 0);
	cudaGraphicsUnregisterResource(cudaVbo);	
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
}

void Cube::draw() {
	static float *proj = new float[16], *view = new float[16];

	//switch program
	glUseProgram(program);

	//load uniform variables
	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, proj);
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, view);
	glUniformMatrix4fv(modelMatrixLocation, 1, GL_TRUE, consts::identity4);

	//just call the vao
	glBindVertexArray(vao);
	glDrawArrays(GL_QUADS, 0, subsize);
	glBindVertexArray(0);

	glUseProgram(0);
}

void Cube::animate() {

	qglviewer::Vec cameraPos = Globals::viewer->camera()->position();
	alListener3f(AL_POSITION, cameraPos.x, cameraPos.y, cameraPos.z);
	//log_console.infoStream() << "CAMERA POS " << cameraPos.x << " " << cameraPos.y << " " << cameraPos.z;

	static int counter = 0;
	static float dx = -0.1f;

	if(counter % 50 == 0)
		dx*=-1;

	//give cuda exclusive access to the vbo
	cudaGraphicsMapResources(1, &cudaVbo, 0);

	//get device data pointer
	float *vertex;
	size_t size;
	cudaGraphicsResourceGetMappedPointer((void**) &vertex, &size, cudaVbo);	

	//compute some random kernel
	moveVertexKernel(vertex, subsize/3, dx);
	cudaDeviceSynchronize();
	checkKernelExecution();

	//give back access to openGL
	cudaGraphicsUnmapResources(1, &cudaVbo, 0);

	counter++;
}

