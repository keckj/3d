
#include "headers.h"
#include "marchingCubes.h"
#include "globals.h"
#include "mc_utils.h"
		
unsigned int MarchingCubes::_triTableUBO = 0;
unsigned int MarchingCubes::_lookupTableUBO = 0;
			
MarchingCubes::MarchingCubes(unsigned int width, unsigned int height, unsigned int length, float voxelSize) :
	_textureWidth(width), _textureHeight(height), _textureLength(length),
	_voxelGridWidth(width-1), _voxelGridHeight(height-1), _voxelGridLength(length-1), 
	_voxelWidth(voxelSize), _voxelHeight(voxelSize), _voxelLength(voxelSize)
{

	if(_triTableUBO == 0 && _lookupTableUBO == 0) {
		generateUniformBlockBuffers();
	}

	//float *data = new float[4*_textureWidth*_textureHeight*_textureLength];
	//int kk = 0;
	//for (unsigned int k = 0; k < _textureLength; k++) {
		//for (unsigned int j = 0; j < _textureHeight; j++) {
			//for (unsigned int i = 0; i < _textureWidth; i++) {
				//kk = k % 3;
				//data[4*(k*_textureHeight*_textureWidth + j*_textureWidth + i)+0] = (kk == 0)*1.0;
				//data[4*(k*_textureHeight*_textureWidth + j*_textureWidth + i)+1] = (kk == 1)*1.0;
				//data[4*(k*_textureHeight*_textureWidth + j*_textureWidth + i)+2] = (kk == 2)*1.0;
				//data[4*(k*_textureHeight*_textureWidth + j*_textureWidth + i)+3] = 1.0;
			//}
		//}
	//}
	
	//_density = new Texture3D(_textureWidth, _textureHeight,_textureLength, GL_RGBA, data, GL_RGBA, GL_FLOAT);
	//_density->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_CLAMP));
	//_density->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_CLAMP));
	//_density->addParameter(Parameter(GL_TEXTURE_WRAP_R, GL_CLAMP));
	//_density->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	//_density->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	//_density->bindAndApplyParameters(0); //create texture and apply parameters
	//dynamic_cast<Texture3D *>(_density)->setData(0);
	
	_density = new Texture3D(_textureWidth, _textureHeight,_textureLength, GL_R16F, 0, GL_RED, GL_FLOAT);
	_density->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_CLAMP));
	_density->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_CLAMP));
	_density->addParameter(Parameter(GL_TEXTURE_WRAP_R, GL_CLAMP));
	_density->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	_density->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	_density->bindAndApplyParameters(0); //create texture and apply parameters

	generateQuads();
	makeDrawProgram();

	generateFullScreenQuad();
	makeDensityProgram();

	generateMarchingCubesPoints();
	makeMarchingCubesProgram();

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
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0,0,_textureWidth,_textureHeight); 

	
	_densityProgram->use();
	glUniform1i(_densityUniformLocs["totalLayers"], _textureLength);
	glUniform2f(_densityUniformLocs["textureSize"], _textureWidth, _textureHeight);

	glBindBuffer(GL_ARRAY_BUFFER, _fullscreenQuadVBO);           
	glEnableVertexAttribArray(0);
	glVertexAttribDivisor(0,0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glDrawArraysInstanced(GL_TRIANGLES, 0, 6, _textureLength);

	glBindBuffer(GL_ARRAY_BUFFER, 0);           
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_DEPTH_TEST);

	glUseProgram(0);
	
	glPopAttrib();

	//create general data uniform block
	GLfloat generalData[12] = {
							(GLfloat)_textureWidth,   (GLfloat)_textureHeight,   (GLfloat)_textureLength,   0.0f,
							(GLfloat)_voxelGridWidth, (GLfloat)_voxelGridHeight, (GLfloat)_voxelGridLength, 0.0f,
							(GLfloat)_voxelWidth,     (GLfloat)_voxelHeight,     (GLfloat)_voxelLength,     0.0f
						};


	glGenBuffers(1, &_generalDataUBO);
	glBindBuffer(GL_UNIFORM_BUFFER, _generalDataUBO);
	glBufferData(GL_UNIFORM_BUFFER, 12*sizeof(GLfloat), generalData, GL_STATIC_DRAW);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

MarchingCubes::~MarchingCubes() {
}

void MarchingCubes::drawDownwards(const float *currentTransformationMatrix) {

	//_drawProgram->use();
	//static float *proj = new float[16], *view = new float[16];
	//glGetFloatv(GL_MODELVIEW_MATRIX, view);
	//glGetFloatv(GL_PROJECTION_MATRIX, proj);
	//glUniformMatrix4fv(_drawUniformLocs["projectionMatrix"], 1, GL_FALSE, proj);
	//glUniformMatrix4fv(_drawUniformLocs["viewMatrix"], 1, GL_FALSE, view);
	//glUniformMatrix4fv(_drawUniformLocs["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);

	//glUniform1i(_drawUniformLocs["layers"], _textureLength);

	//glBindBuffer(GL_ARRAY_BUFFER, _vertexVBO);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	//glVertexAttribDivisor(0,0);
	//glEnableVertexAttribArray(0);

	//glDrawArraysInstanced(GL_QUADS, 0, 4, _textureLength);

	//glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glUseProgram(0);
	
	
	_marchingCubesProgram->use();
	
	static float *proj = new float[16], *view = new float[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, view);
	glGetFloatv(GL_PROJECTION_MATRIX, proj);
	glUniformMatrix4fv(_marchingCubesUniformLocs["projectionMatrix"], 1, GL_FALSE, proj);
	glUniformMatrix4fv(_marchingCubesUniformLocs["viewMatrix"], 1, GL_FALSE, view);
	
	glBindBufferBase(GL_UNIFORM_BUFFER, 0 , _lookupTableUBO);
	glBindBufferBase(GL_UNIFORM_BUFFER, 1 , _triTableUBO);
	glBindBufferBase(GL_UNIFORM_BUFFER, 2 , _generalDataUBO);

	glBindBuffer(GL_ARRAY_BUFFER, _marchingCubesLowerLeftXY_VBO);           
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(0,0);
	glEnableVertexAttribArray(0);

	glDrawArraysInstanced(GL_POINTS, 0, _voxelGridWidth*_voxelGridHeight, _voxelGridLength);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, 0);
	glUseProgram(0);
}

void MarchingCubes::generateQuads() {

	float quads[4*3] = {0, 0, 0,
		0, 1, 0,
		1, 1, 0,
		1, 0, 0};

	glGenBuffers(1, &_vertexVBO);
	glBindBuffer(GL_ARRAY_BUFFER, _vertexVBO);
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

	glGenBuffers(1, &_fullscreenQuadVBO);
	glBindBuffer(GL_ARRAY_BUFFER, _fullscreenQuadVBO);
	glBufferData(GL_ARRAY_BUFFER, 6*3*sizeof(float), buffer, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

}
		
void MarchingCubes::generateMarchingCubesPoints() {
	
	float *lowerLeftXY = new float[2*_voxelGridWidth*_voxelGridHeight];

	float stepX = 1.0f/_voxelGridWidth;
	float stepY = 1.0f/_voxelGridHeight;

	float posY = 0;
	for (unsigned int j = 0; j < _voxelGridHeight; j++) {
		float posX = 0;
		for (unsigned int i = 0; i < _voxelGridWidth; i++) {
			lowerLeftXY[2*(j*_voxelGridWidth + i) + 0] = posX;
			lowerLeftXY[2*(j*_voxelGridWidth + i) + 1] = posY;
			posX += stepX;
		}
		posY += stepY;
	}

	glGenBuffers(1, &_marchingCubesLowerLeftXY_VBO);
	glBindBuffer(GL_ARRAY_BUFFER, _marchingCubesLowerLeftXY_VBO);
	glBufferData(GL_ARRAY_BUFFER, 2*_voxelGridWidth*_voxelGridHeight*sizeof(float), lowerLeftXY, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	delete [] lowerLeftXY;
}

void MarchingCubes::makeDrawProgram() {

	_drawProgram = new Program("Draw test");
	_drawProgram->bindAttribLocations("0", "vertex_position");
	_drawProgram->bindFragDataLocation(0, "out_colour");

	_drawProgram->attachShader(Shader("shaders/marchingCubes/test_vs.glsl", GL_VERTEX_SHADER));
	_drawProgram->attachShader(Shader("shaders/marchingCubes/test_fs.glsl", GL_FRAGMENT_SHADER));

	_drawProgram->link();
	_drawUniformLocs = _drawProgram->getUniformLocationsMap("modelMatrix projectionMatrix viewMatrix layers", true);

	_drawProgram->bindTextures(&_density, "density", true);
}

void MarchingCubes::makeDensityProgram() {
	_densityProgram = new Program("Density");
	_densityProgram->bindAttribLocations("0", "vertex_position");
	_densityProgram->bindFragDataLocation(0, "out_colour");

	_densityProgram->attachShader(Shader("shaders/marchingCubes/density_vs.glsl", GL_VERTEX_SHADER));
	_densityProgram->attachShader(Shader("shaders/marchingCubes/density_gs.glsl", GL_GEOMETRY_SHADER));
	_densityProgram->attachShader(Shader("shaders/marchingCubes/density_fs.glsl", GL_FRAGMENT_SHADER));
	
	_densityProgram->link();
	_densityUniformLocs = _densityProgram->getUniformLocationsMap("totalLayers textureSize", false);
}

void MarchingCubes::makeMarchingCubesProgram() {
	_marchingCubesProgram = new Program("Marching Cube");
	_marchingCubesProgram->bindAttribLocations("0", "voxelLowerLeftXY");
	_marchingCubesProgram->bindUniformBufferLocations("0 1 2", "lookupTable triangleTable generalData");

	_marchingCubesProgram->attachShader(Shader("shaders/marchingCubes/marchingCube_vs.glsl", GL_VERTEX_SHADER));
	_marchingCubesProgram->attachShader(Shader("shaders/marchingCubes/marchingCube_gs.glsl", GL_GEOMETRY_SHADER));
	_marchingCubesProgram->attachShader(Shader("shaders/marchingCubes/marchingCube_fs.glsl", GL_FRAGMENT_SHADER));

	_marchingCubesProgram->link();

	_marchingCubesUniformLocs = _marchingCubesProgram->getUniformLocationsMap("viewMatrix projectionMatrix", true);
	
	_marchingCubesProgram->bindTextures(&_density, "density", false);
}

void MarchingCubes::generateUniformBlockBuffers() {
		
	log_console.infoStream() << "Size of GLbyte : " << sizeof(GLbyte);
	log_console.infoStream() << "Size of GLfloat : " << sizeof(GLfloat);
	log_console.infoStream() << "Size of GLint : " << sizeof(GLint);
	log_console.infoStream() << "Size of GLuint : " << sizeof(GLuint);
	log_console.infoStream() << "Generating static marching cubes uniform block buffers !";
	
	glGenBuffers(1, &_triTableUBO);
	glBindBuffer(GL_UNIFORM_BUFFER, _triTableUBO);
	glBufferData(GL_UNIFORM_BUFFER, 5120*sizeof(GLfloat), MarchingCube::triangleTable, GL_STATIC_DRAW);
	
	glGenBuffers(1, &_lookupTableUBO);
	glBindBuffer(GL_UNIFORM_BUFFER, _lookupTableUBO);

	glBufferData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 48*6*sizeof(GLfloat), 0, GL_STATIC_DRAW);
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 1024*sizeof(GLuint), MarchingCube::caseToNumPoly);
	glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 0*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::edgeStart);
	glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 1*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::edgeDir);
	glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 2*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::maskA0123);
	glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 3*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::maskB0123);
	glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 4*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::maskA4567);
	glBufferSubData(GL_UNIFORM_BUFFER, 1024*sizeof(GLuint) + 5*48*sizeof(GLfloat), 48*sizeof(GLfloat), MarchingCube::maskB4567);
	
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}
