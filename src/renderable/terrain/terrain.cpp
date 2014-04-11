

#include <cassert>
#include <cstring>

#include "log.h"
#include "terrain.h"
#include <GL/glew.h>

Terrain::Terrain(unsigned char *heightmap, unsigned int width, unsigned int height, bool centered) :
	program(0), textures(0), VAO(0), VBO(0),
	width(width), height(height), centered(centered)
{
	initializeRelativeModelMatrix();

	makeProgram();

	log_console.infoStream() << "Generating a " << width << "x" << height << " terrain with program " << program->getProgramId() << " !";

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
			writeVec3f(colors, coloridx, alpha*x,alpha*y,0.0f);		
			writeVec3f(colors, coloridx, alpha*x,alpha*(y-1),0.0f);		
		}

		writeVec3f(vertex, idx, width-1, y-1,heightmap[y*width-1]);		
		writeVec3f(vertex, idx, 0, y-1,heightmap[(y-1)*width]);		
		writeVec3f(colors, coloridx, 0.0f,0.0f,0.0f);		
		writeVec3f(colors, coloridx, 0.0f,0.0f,0.0f);		
	}

	
	sendToDevice();


	delete [] vertex;
	delete [] colors;

	vertex = 0;
	colors = 0;
}

Terrain::~Terrain() {
	for (int i = 0; i < 5; i++) {
		delete textures[i];
	}

	delete [] textures;
	delete program;
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
	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 0, (char*)NULL + size/2);
	glEnableVertexAttribArray(5);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}


void Terrain::drawDownwards(const float *currentTransformationMatrix) {
	static float *proj = new float[16], *view = new float[16];

	program->use();

	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glUniformMatrix4fv(uniformLocs["projectionMatrix"], 1, GL_FALSE, proj);
	glUniformMatrix4fv(uniformLocs["viewMatrix"], 1, GL_FALSE, view);
	glUniformMatrix4fv(uniformLocs["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);
	
	glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, nVertex);
	glBindVertexArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
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

void Terrain::initializeRelativeModelMatrix() {
	
	float alpha = 0.05f;
	float beta = 0.05f;
	float gamma = 0.03f;
	const float scale[] = {
		alpha, 0.0f, 0.0f, centered ? -alpha*width/2.0f : 0.0f,
		0.0f, beta, 0.0f, centered ? -beta*height/2.0f : 0.0f,
		0.0f, 0.0f, gamma, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	};

	setRelativeModelMatrix(scale);
}

void Terrain::makeProgram() {

		textures = new Texture*[5];
        
		textures[0] = new Texture2D("textures/forest 13.png","png");
		textures[0]->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_REPEAT));
		textures[0]->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_REPEAT));
		textures[0]->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
		textures[0]->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
		textures[0]->generateMipMap();

		textures[1] = new Texture2D("textures/grass 9.png", "png");
		textures[1]->addParameters(textures[0]->getParameters());
		textures[1]->generateMipMap();
		
		textures[2] = new Texture2D("textures/grass 7.png", "png");
		textures[2]->addParameters(textures[0]->getParameters());
		textures[2]->generateMipMap();

		textures[3] = new Texture2D("textures/dirt 4.png", "png");
		textures[3]->addParameters(textures[0]->getParameters());
		textures[3]->generateMipMap();
		
		textures[4] = new Texture2D("textures/snow 1.png", "png");
		textures[4]->addParameters(textures[0]->getParameters());
		textures[4]->generateMipMap();


        program = new Program("Terrain poulpy");

        program->bindAttribLocations("4 5", "vertex_position vertex_colour");
        program->bindFragDataLocation(0, "out_colour");

        program->attachShader(Shader("shaders/terrain/vs.glsl", GL_VERTEX_SHADER));
        program->attachShader(Shader("shaders/terrain/fs.glsl", GL_FRAGMENT_SHADER));

        program->link();
		
        uniformLocs = program->getUniformLocationsMap("modelMatrix projectionMatrix viewMatrix", true);
	
		program->bindTextures(textures, "texture_1 texture_2 texture_3 texture_4 texture_5", true);
}
