
#include "terrain.h"
#include <cassert>
#include <cstring>
#include "log.h"

#include <GL/glew.h>

Terrain::Terrain(unsigned char *heightmap, unsigned int width, unsigned int height, bool centered, 
			unsigned int program,
			unsigned int modelMatrixLocation, unsigned int projectionMatrixLocation, unsigned int viewMatrixLocation) :
	width(width), height(height), centered(centered), 
	program(program), 
	modelMatrixLocation(modelMatrixLocation), projectionMatrixLocation(projectionMatrixLocation), viewMatrixLocation(viewMatrixLocation),
	VAO(0), VBO(0)
{

	assert(glIsProgram(program));

	log_console.infoStream() << "Generating a " << width << "x" << height << "terrain with program " << program << " !";

	nVertex = 2*(height-1)*(width+1); 
	vertex = new float[3*nVertex];
	colors = new float[3*nVertex];

	unsigned int idx = 0;
	unsigned int coloridx = 0;
	float alpha = 0.01;
	for (unsigned int y = height-1; y > 0; y--) {

		for (unsigned int x = 0; x < width; x++) {
			writeVec3f(vertex, idx, x,y,heightmap[y*width+x]);		
			writeVec3f(vertex, idx, x, y-1,heightmap[(y-1)*width+x]);		
			//writeColor(heightmap[y*width+x], coloridx, colors);
			//writeColor(heightmap[(y-1)*width+x], coloridx, colors);
			writeVec3f(colors, coloridx, alpha*x,alpha*y,0.0f);		
			writeVec3f(colors, coloridx, alpha*x,alpha*(y-1),0.0f);		
		}

		writeVec3f(vertex, idx, width-1, y-1,heightmap[y*width-1]);		
		writeVec3f(vertex, idx, 0, y-1,heightmap[(y-1)*width]);		
		//writeColor(heightmap[y*width - 1], coloridx, colors);
		//writeColor(heightmap[(y-1)*width], coloridx, colors);
		writeVec3f(colors, coloridx, 0.0f,0.0f,0.0f);		
		writeVec3f(colors, coloridx, 0.0f,0.0f,0.0f);		

	}

	sendToDevice();

	delete [] vertex;
	delete [] colors;

	vertex = 0;
	colors = 0;
}

void inline Terrain::sendToDevice() {

	unsigned int size = 2*3*nVertex*sizeof(float);

	if(glIsBuffer(VBO)) 
		glDeleteBuffers(1, &VBO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);	
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, size/2, vertex);
	glBufferSubData(GL_ARRAY_BUFFER, size/2, size/2, colors);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	if (glIsVertexArray(VAO))
		glDeleteVertexArrays(1, &VAO);	

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (char*)NULL + size/2);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}


Terrain::~Terrain() {
}

void Terrain::draw() {
	static float *proj = new float[16], *view = new float[16];

	log_console.infoStream() << "Draw ! " << program << " " << modelMatrixLocation ;
	glUseProgram(program);
	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, proj); //true
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, view); //false
	glUniformMatrix4fv(modelMatrixLocation, 1, GL_TRUE, getRelativeModelMatrix());
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, nVertex);
	glBindVertexArray(0);
	glUseProgram(0);

}

void Terrain::writeColor(int height, unsigned int &idx, float *color) {

	float val = height/255.0f;
	color[idx++] = val;
	color[idx++] = val;
	color[idx++] = val;
}

void Terrain::writeVec3f(float *array, unsigned int &idx, float x, float y, float z) {
	array[idx++] = x;
	array[idx++] = y;
	array[idx++] = z;
}

const float *Terrain::getRelativeModelMatrix() const {
	
	float alpha = 0.05f;
	float beta = 0.05f;
	float gamma = 0.03f;
	const float scale[] = {
		alpha, 0.0f, 0.0f, centered ? -alpha*width/2.0f : 0.0f,
		0.0f, beta, 0.0f, centered ? -beta*height/2.0f : 0.0f,
		0.0f, 0.0f, gamma, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	};

	float *scaleCpy = new float[16];
	memcpy(scaleCpy, scale, 16*sizeof(float));

	return scaleCpy;
}
