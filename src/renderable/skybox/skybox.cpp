#include "headers.h"
#include "skybox.h"
#include "globals.h"
	
bool Skybox::_init = false;
unsigned int Skybox::_vertexVBO = 0;

float Skybox::_vertexCoords[] {
		1.0f, -1.0f, -1.0f,
		1.0f, -1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		
		-1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,


		-1.0f,  1.0f, -1.0f,
		1.0f,  1.0f, -1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f, -1.0f,
		
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		1.0f, -1.0f,  1.0f,
		
		
		-1.0f, -1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f,  1.0f,  1.0f,
		1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,
		
		-1.0f,  1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,
		1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f
};
		
void Skybox::initVBOs() {
	glGenBuffers(1, &_vertexVBO);
	glBindBuffer(GL_ARRAY_BUFFER, _vertexVBO);	
	glBufferData(GL_ARRAY_BUFFER, 6*2*3*3*sizeof(float), _vertexCoords, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);	
	
	_init = true;
}

Skybox::Skybox(const std::string &folder, const std::string &fileNames, const std::string &format) {

	if(!_init)
		initVBOs();

	_cubeMap = new CubeMap(folder, fileNames, format);
	_cubeMap->addParameter(Parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	_cubeMap->addParameter(Parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	_cubeMap->addParameter(Parameter(GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
	_cubeMap->addParameter(Parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	_cubeMap->addParameter(Parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	_cubeMap->generateMipMap();

	makeProgram();

	log_console.infoStream() << "Created skybox from folder " << folder << " with files " 
		<< fileNames << " !";
}

Skybox::~Skybox () {
	delete _cubeMap;
	delete _program;
}

void Skybox::drawDownwards(const float *currentTransformationMatrix) {

	_program->use();
	
	glPushAttrib(GL_DEPTH_BUFFER_BIT);
	glPushAttrib(GL_POLYGON_BIT);

	glCullFace(GL_BACK);
	glDisable(GL_DEPTH_TEST);
	
	glEnable(GL_TEXTURE_CUBE_MAP);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

	glBindBufferBase(GL_UNIFORM_BUFFER, 0,  Globals::projectionViewUniformBlock);
	glUniformMatrix4fv(_uniformLocations["modelMatrix"], 1, GL_TRUE, currentTransformationMatrix);

	glBindBuffer(GL_ARRAY_BUFFER, _vertexVBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	
	glDrawArrays(GL_TRIANGLES, 0, 6*2*3);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glUseProgram(0);

	glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glDisable(GL_TEXTURE_CUBE_MAP);

	glPopAttrib();
	glPopAttrib();
}

void Skybox::makeProgram() {
        _program = new Program("Skybox");

        _program->bindAttribLocations("0", "vertex_position");
        _program->bindFragDataLocation(0, "out_colour");
        _program->bindUniformBufferLocations("0", "projectionView");

        _program->attachShader(Shader("shaders/skybox/vs.glsl", GL_VERTEX_SHADER));
        _program->attachShader(Shader("shaders/skybox/fs.glsl", GL_FRAGMENT_SHADER));

        _program->link();
		
        _uniformLocations = _program->getUniformLocationsMap("modelMatrix", true);
	
		_program->bindTextures(&_cubeMap, "cubemap", true);
}

Texture* Skybox::getCubeMap() {
    return _cubeMap; 
}

