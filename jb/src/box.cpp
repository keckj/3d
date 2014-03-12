#include "box.hpp"
#include "log.hpp"
#include <cstring>

Box::Box(float width, float height, float length, 
		float x, float y, float z, bool centered,
		unsigned int program, int modelMatrixLocation)
: 
	width(width), height(height), length(length), 
	x(x), y(y), z(z), centered(centered), 
	program(program), modelMatrixLocation(modelMatrixLocation),
	VAO(0), VBO(0), vertex(0), colors(0)
{
	vertex = new float[6*4*3];
	normals = new float[6*4*3];
	colors = new float[6*4*3];
	
	computeVertex();

	sendToDevice();

	delete [] vertex;
	delete [] normals;
	delete [] colors;

	vertex = 0;
	normals = 0;;
	colors = 0;
}
		
Box::~Box() {
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);	
}

void Box::setPosition(float x, float y, float z) {
	this->x = x;
	this->y = y;
	this->z = z;
}

void Box::draw(const float *modelMatrix) const {
	log_console.infoStream() << "DRAW using program " << program 
		<< " with modelMatrix at ID " << modelMatrixLocation;
	
	for (int i = 0; i < 4; i++) {
		log_console.infoStream() 
			<< "\t" << modelMatrix[4*i+0] 
			<< "\t" << modelMatrix[4*i+1] 
			<< "\t" << modelMatrix[4*i+2] 
			<< "\t" << modelMatrix[4*i+3];
	}

	glUseProgram(program);
		glUniformMatrix4fv(modelMatrixLocation, 1, GL_TRUE, modelMatrix);
		glBindVertexArray(VAO);
			glDrawArrays(GL_QUADS, 0, 4*6);
		glBindVertexArray(0);
	glUseProgram(0);
}
		
void inline Box::computeVertex() {
	
	unsigned int idx = 0;	
	for (int fx = 0; fx <= 1; fx++) {
		writeVec3f(vertex, idx, x+fx*width, y+0.0f, z+0.0f);
		writeVec3f(vertex, idx, x+fx*width, y+height, z+0.0f);
		writeVec3f(vertex, idx, x+fx*width, y+height, z+length);
		writeVec3f(vertex, idx, x+fx*width, y+0.0f, z+length);
	}
	for (int fy = 0; fy <= 1; fy++) {
		writeVec3f(vertex, idx, x+0.0f, y+fy*height, z+0.0f);
		writeVec3f(vertex, idx, x+width, y+fy*height, z+0.0f);
		writeVec3f(vertex, idx, x+width, y+fy*height, z+length);
		writeVec3f(vertex, idx, x+0.0f, y+fy*height, z+length);
	}
	for (int fz = 0; fz <= 1; fz++) {
		writeVec3f(vertex, idx, x+0.0f, y+0.0f, z+fz*length);
		writeVec3f(vertex, idx, x+width, y+0.0f, z+fz*length);
		writeVec3f(vertex, idx, x+width, y+height, z+fz*length);
		writeVec3f(vertex, idx, x+0.0f, y+height, z+fz*length);
	}

	idx = 0;	
	for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 4; j++) {
				switch(i/2) {
					case 0:
						writeVec3f(colors, idx, 1.0f, 1.0f, 0.0f);
						break;
					case 1:
						writeVec3f(colors, idx, 0.0f, 1.0f, 1.0f);
						break;
					case 2:
						writeVec3f(colors, idx, 1.0f, 0.0f, 1.0f);
						break;
				}
			}
	}

	idx = 0;
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 4; j++) {
			switch(i/2) {
				case 0:
					writeVec3f(normals, idx, i%2==0 ? -1.0f : 1.0f, 0.0f, 0.0f);
					break;
				case 1:
					writeVec3f(normals, idx, 0.0f, i%2==0 ? -1.0f : 1.0f, 0.0f);
					break;
				case 2:
					writeVec3f(normals, idx, 0.0f, 0.0f, i%2==0 ? -1.0f : 1.0f);
					break;
			}
		}
	}
	
}
		
void inline Box::sendToDevice() {
	
	unsigned int size = 3*6*4*3*sizeof(float);
	
	if (glIsBuffer(VBO)) 
		glDeleteBuffers(1, &VBO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);	
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, size/3, vertex);
	glBufferSubData(GL_ARRAY_BUFFER, size/3, size/3, colors);
	glBufferSubData(GL_ARRAY_BUFFER, 2*size/3, size/3, normals);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	
	if (glIsVertexArray(VAO))
		glDeleteVertexArrays(1, &VAO);	
	
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (char*)NULL + size/3);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (char*)NULL + 2*size/3);
			glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	
	
}
		
const float *Box::getRelativeModelMatrix() const {
	
	float lambda = 0.1f;
	const float scale[] = {
			lambda, 0.0f, 0.0f, centered ? -lambda*width/2.0f : 0.0f,
			0.0f, lambda, 0.0f, centered ? -lambda*height/2.0f : 0.0f,
			0.0f, 0.0f, lambda, centered ? -lambda*length/2.0f : 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
	};

	float *scaleCpy = new float[16];
	memcpy(scaleCpy, scale, 16*sizeof(float));

	return scaleCpy;
}
		
void Box::writeVec3f(float *array, unsigned int &idx, float x, float y, float z) {
	array[idx++] = x;
	array[idx++] = y;
	array[idx++] = z;
}
