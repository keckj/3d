
#include "program.h"
#include "globals.h"
#include "shader.h"
#include "log.h"

#include <GL/glew.h>
#include <sstream>
#include <cassert>

Program::Program(std::string const &name) {
	linked = false;
	programId = glCreateProgram();
	programName = name;

	std::stringstream ss;
	ss << "[Program " << programName << "][id=" << programId << "]  ";
	logProgramHead = ss.str();

	log_console.infoStream() << logProgramHead << "Created program " << name << " with ID " << programId << " !";
} 

Program::~Program() {
	glDeleteProgram(programId);
} 

void Program::attachShader(Shader const &shader) {
	glAttachShader(programId, shader.getShader());
	log_console.infoStream() << logProgramHead 
		<<  "Attached shader " << shader.toStringShaderType() << "[id=" << shader.getShader()
		<< "] from file " << shader.getLocation();
}

void Program::bindAttribLocation(unsigned int location, std::string const &attribVarName) {
	log_console.infoStream() << logProgramHead << "Binding attribute '" << attribVarName << "' to location " << location << ".";

	if(location > (unsigned int)Globals::glMaxVertexAttribs) {
		log_console.warnStream() << "Location " << location 
			<< " is superior to GL_MAX_VERTEX_ATTRIBS = " << Globals::glMaxVertexAttribs
			<< "!";
	}

	attribLocations.insert(std::pair<std::string,unsigned int>(attribVarName, location));

	glBindAttribLocation(programId, location, attribVarName.c_str());
}

void Program::bindFragDataLocation(unsigned int location, std::string const &fragVarName) {
	log_console.infoStream() << logProgramHead << "Binding frag data '" << fragVarName << "' to location " << location << ".";

	if(location > (unsigned int)Globals::glMaxDrawBuffers) {
		log_console.warnStream() << "Location " << location 
			<< " is superior to GL_MAX_DRAW_BUFFERS = " << Globals::glMaxDrawBuffers
			<< "!";
	}

	glBindFragDataLocation(programId, location, fragVarName.c_str());
}

void Program::bindAttribLocations(std::string locations, std::string const &attribVarNames) {
	std::stringstream locs(locations);
	std::stringstream names(attribVarNames);

	unsigned int l;
	std::string n;

	while(locs.good() && names.good()) {
		names >> n;
		locs >> l;

		bindAttribLocation(l, n);
	}

	assert(locs.eof());
	assert(names.eof());
}

void Program::bindFragDataLocations(std::string locations, std::string const &fragVarNames) {
	std::stringstream locs(locations);
	std::stringstream names(fragVarNames);

	unsigned int l;
	std::string n;

	while(locs.good() && names.good()) {
		names >> n;
		locs >> l;

		bindFragDataLocation(l, n);
	}

	assert(locs.eof());
	assert(names.eof());
}

//link the program and check for compilation errors
//can be called multiple times Program::(transform feedback)
void Program::link() {
	glLinkProgram(programId);

	int status;
	glGetProgramiv(programId, GL_LINK_STATUS, &status);
	if(status) {
		log_console.infoStream() << logProgramHead << "Linking program ... Success !";
	}
	else {
		log_console.errorStream() << logProgramHead << "Linking program ... Failed !";

		GLchar errorLog[1024] = {0};
		glGetProgramInfoLog(programId, 1024, NULL, errorLog);
		
		log_console.errorStream() << logProgramHead << "Shader compilation log :\n" << errorLog;
		exit(1);
	}

	//check post compilation attrib locations
	std::map<std::string,unsigned int>::const_iterator it = attribLocations.begin();
	for(; it != attribLocations.end(); it++) {
		int id = glGetAttribLocation(programId, it->first.c_str());

		if(id != (int)it->second)
			log_console.warnStream() << logProgramHead <<"Attrib data '" << it->first << "' was not set to location " << it->second << " (id=-1)."; 
	}

	linked = true;
}

//bind program and textures
void Program::use() const {

	if(!linked) {
		log_console.errorStream() << logProgramHead << "Trying to use a program that has not been linked !";
		std::cout << std::flush;
		exit(0);
	}

	glUseProgram(programId);

	std::vector<unsigned int> availableTextureLocations = Texture::requestTextures(linkedTextures.size());
	std::vector<unsigned int>::iterator av_loc_it = availableTextureLocations.begin();

	std::vector<std::pair<int, Texture*> >::const_iterator tex_it = linkedTextures.begin();

	for (; tex_it != linkedTextures.end(); ++tex_it) {
		
		if(tex_it->second->isBinded()) {
			glUniform1i(tex_it->first, tex_it->second->getLastKnownLocation());
			log_console.debugStream() << logProgramHead << "Update uniform location " << tex_it->first << " with value " << tex_it->second->getLastKnownLocation() << ".";
			continue;
		}

		tex_it->second->bindAndApplyParameters(*av_loc_it);
		glUniform1i(tex_it->first, *av_loc_it);
			log_console.debugStream() << logProgramHead << "Update uniform location " << tex_it->first << " with value " << *av_loc_it << ".";
		av_loc_it++;
	}

}

unsigned int Program::getProgramId() const {
	return this->programId;
}

const std::vector<int> Program::getUniformLocations(std::string const &varNames, bool assert) {
	std::vector<int> ids;
	std::string var;

	if(!linked) {
		log_console.errorStream() << logProgramHead << "Trying to get uniform locations in a program that has not been linked !";
		std::cout << std::flush;
		exit(0);
	}

	std::stringstream ss(varNames);

	glUseProgram(programId);
	while(ss.good()) {
		ss >> var;	
		int id = glGetUniformLocation(programId, var.c_str());

		if(id == -1) {
			if(assert) {
				log_console.errorStream() << logProgramHead << "Uniform variable location of '" << var <<"' is -1 !";		
				std::cout << std::flush;
				exit(1);
			}
			else {
				log_console.warnStream() << logProgramHead << "Uniform variable location of '" << var <<"' is -1 !";		
			}
		}

		ids.push_back(id);
	}
	glUseProgram(0);

	return ids;
} 

const std::map<std::string,int> Program::getUniformLocationsMap(std::string const &varNames, bool assert) { //separated by space
	std::map<std::string,int> map;

	std::stringstream ss(varNames);
	std::string var;

	if(!linked) {
		log_console.errorStream() << logProgramHead << "Trying to get uniform locations in a program that has not been linked !";
		std::cout << std::flush;
		exit(0);
	}

	glUseProgram(programId);

	while(ss.good()) {
		ss >> var;	
		int id = glGetUniformLocation(programId, var.c_str());

		if(id == -1) {
			if(assert) {
				log_console.errorStream() << logProgramHead << "Uniform variable location of '" << var <<"' is -1 !";		
				std::cout << std::flush;
				exit(1);
			} 
			else {
				log_console.warnStream() << logProgramHead << "Uniform variable location of '" << var <<"' is -1 !";		
			}
		}

		map.insert(std::pair<std::string,unsigned int>(var, id));
	}

	glUseProgram(0);

	return map;
}

void Program::bindTextures(Texture **textures, std::string uniformNames, bool assert) {

	std::vector<int> locations = getUniformLocations(uniformNames, assert);

	std::vector<int>::iterator it = locations.begin();
	int i = 0;
	for (; it != locations.end(); ++it) {
		if(*it == -1)
			continue;

		linkedTextures.push_back(std::pair<int, Texture*>(*it, textures[i++]));
	}
}

void Program::resetDefaultGlProgramState() {
	glUseProgram(0);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D,0);

	//ENABLE
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_POLYGON_SMOOTH);

	//DISABLE	
	glDisable(GL_TEXTURE_1D);
	glDisable(GL_TEXTURE_2D);

	glDisable(GL_BLEND);
	glDisable(GL_FOG);
	glDisable(GL_ALPHA_TEST);

	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHT1);
	glDisable(GL_LIGHT2);
	glDisable(GL_LIGHT3);
	glDisable(GL_LIGHT4);
	glDisable(GL_LIGHT5);
	glDisable(GL_LIGHT6);
	glDisable(GL_LIGHT7);

	glDisable(GL_AUTO_NORMAL);
	glDisable(GL_NORMALIZE);
	glDisable(GL_DITHER);
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_LINE_STIPPLE);
	glDisable(GL_POLYGON_STIPPLE);
	glDisable(GL_LOGIC_OP);
}
