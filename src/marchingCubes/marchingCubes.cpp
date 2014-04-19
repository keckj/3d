
#include "headers.h"
#include "marchingCubes.h"
#include "globals.h"
		
MarchingCubes::MarchingCubes(unsigned int width, unsigned int height, unsigned int length) :
	_width(width), _height(height), _length(length)
{

	float *data = new float[4*_width*_height*_length];

	int kk = 0;
	for (unsigned int k = 0; k < _length; k++) {
		for (unsigned int j = 0; j < _height; j++) {
			for (unsigned int i = 0; i < _width; i++) {
				kk = k % 3;
				data[4*(k*_height*_width + j*_width + i)+0] = (kk == 0)*1.0;
				data[4*(k*_height*_width + j*_width + i)+1] = (kk == 1)*1.0;
				data[4*(k*_height*_width + j*_width + i)+2] = (kk == 2)*1.0;
				data[4*(k*_height*_width + j*_width + i)+3] = 1.0;
			}
		}
	}
	
	_density = new Texture3D(_width, _height,_length, GL_RGBA, data, GL_RGBA, GL_FLOAT);
	_density->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_CLAMP));
	_density->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_CLAMP));
	_density->addParameter(Parameter(GL_TEXTURE_WRAP_R, GL_CLAMP));
	_density->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	_density->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	_density->bindAndApplyParameters(0); //create texture and apply parameters
	dynamic_cast<Texture3D *>(_density)->setData(0);

	generateQuads();
	makeDrawProgram();

	generateFullScreenQuad();
	makeDensityProgram();

	//The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
	GLuint frameBuffer = 0;
	glGenFramebuffers(1, &frameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _density->getTextureId(), 0); //level 0 

	// Set the list of draw buffers.
	static const GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

	//Always check that our framebuffer is ok
	switch(glCheckFramebufferStatus(GL_FRAMEBUFFER))  {
		case GL_FRAMEBUFFER_COMPLETE:
			log_console.infoStream() << "Framebuffer complete !";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
			log_console.errorStream() << "Framebuffer incomplete layer targets !";
			exit(1);
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			log_console.errorStream() << "Framebuffer incomplete attachement !";
			exit(1);
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			log_console.errorStream() << "Framebuffer missing attachment !";
			exit(1);
			break;
		case GL_FRAMEBUFFER_UNSUPPORTED:
			log_console.errorStream() << "Framebuffer unsupported !";
			exit(1);
			break;
		default:
			log_console.errorStream() << "Something went wrong when configuring the framebuffer !";
			exit(1);
	}

	// Render on the whole framebuffer, complete from the lower left corner to the upper right
	int viewport[4];
	Globals::viewer->camera()->getViewport(viewport);
	glViewport(viewport[0],viewport[1],viewport[2], viewport[3]); 

	
	_densityProgram->use();
	glUniform1i(_densityUniformLocs["totalLayers"], _length);

	glBindBuffer(GL_ARRAY_BUFFER, fullscreenQuadVBO);           
	glEnableVertexAttribArray(0);
	glVertexAttribDivisor(0,0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glDrawArraysInstanced(GL_TRIANGLES, 0, 6, _length);

	glBindBuffer(GL_ARRAY_BUFFER, 0);           
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_DEPTH_TEST);

	glUseProgram(0);
	
}

MarchingCubes::~MarchingCubes() {
}

void MarchingCubes::drawDownwards(const float *currentTransformationMatrix) {

	_drawProgram->use();

	static float *proj = new float[16], *view = new float[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glUniformMatrix4fv(_drawUniformLocs["projectionMatrix"], 1, GL_FALSE, proj);
	glUniformMatrix4fv(_drawUniformLocs["viewMatrix"], 1, GL_FALSE, view);
	glUniformMatrix4fv(_drawUniformLocs["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);

	glUniform1i(_drawUniformLocs["layers"], _length);

	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(0,0);
	glEnableVertexAttribArray(0);

	glDrawArraysInstanced(GL_QUADS, 0, 4, _length);
	//glPointSize(10);
	//glDrawArrays(GL_QUADS, 0, 4);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glUseProgram(0);
}

void MarchingCubes::generateQuads() {

	float quads[4*3] = {0, 0, 0,
		0, 1, 0,
		1, 1, 0,
		1, 0, 0};

	glGenBuffers(1, &vertexVBO);
	glBindBuffer(GL_ARRAY_BUFFER, vertexVBO);
	glBufferData(GL_ARRAY_BUFFER, 4*3*sizeof(float), quads, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void MarchingCubes::generateFullScreenQuad() {
	float buffer[] = {
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,

		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,

		//0.0f, 0.0f,
		//1.0f, 0.0f,
		//1.0f, 1.0f,

		//0.0f, 1.0f,
		//0.0f, 0.0f,
		//1.0f, 1.0f
	};

	glGenBuffers(1, &fullscreenQuadVBO);
	glBindBuffer(GL_ARRAY_BUFFER, fullscreenQuadVBO);
	glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(float), buffer, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void MarchingCubes::makeDrawProgram() {

	_drawProgram = new Program("Test Program");
	_drawProgram->bindAttribLocations("0", "vertex_position");
	_drawProgram->bindFragDataLocation(0, "out_colour");

	_drawProgram->attachShader(Shader("shaders/marchingCubes/test_vs.glsl", GL_VERTEX_SHADER));
	_drawProgram->attachShader(Shader("shaders/marchingCubes/test_fs.glsl", GL_FRAGMENT_SHADER));

	_drawProgram->link();
	_drawUniformLocs = _drawProgram->getUniformLocationsMap("modelMatrix projectionMatrix viewMatrix layers", true);

	_drawProgram->bindTextures(&_density, "density", true);
}

void MarchingCubes::makeDensityProgram() {
	_densityProgram = new Program("Density Program");
	_densityProgram->bindAttribLocations("0", "vertex_position");
	_densityProgram->bindAttribLocations("1", "tex_coord");
	_densityProgram->bindFragDataLocation(0, "out_colour");

	_densityProgram->attachShader(Shader("shaders/marchingCubes/density_vs.glsl", GL_VERTEX_SHADER));
	_densityProgram->attachShader(Shader("shaders/marchingCubes/density_gs.glsl", GL_GEOMETRY_SHADER));
	_densityProgram->attachShader(Shader("shaders/marchingCubes/density_fs.glsl", GL_FRAGMENT_SHADER));
	
	_densityProgram->link();
	_densityUniformLocs = _densityProgram->getUniformLocationsMap("totalLayers boxSize", false);
}


