
#include "shader.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>

using namespace std;

Shader::Shader(const char* location, GLenum shaderType) {

	// Load shader file
	stringstream shaderString;
	string line;

	ifstream shaderFile(location);
	
	if (shaderFile.is_open() && getline(shaderFile, line))
	{
		shaderString << line;

		while (getline(shaderFile,line))
			shaderString << "\n" << line;

		shaderFile.close();

		clog << "\nLoading shader from file " << location;
		clog << "\n-- SHADER -- \n" << shaderString.str() << "\n-- END --";
	}

	else {
		clog << "\nUnable to load shader from file " << location; 
	}
	
	// Create and compile shader
	Shader::shaderType = shaderType;
	
	Shader::shader = glCreateShader(shaderType);
	
	const string prog_string = shaderString.str();
	const GLint prog_length = prog_string.length();
	const char* prog = prog_string.c_str();

	glShaderSource(shader, 1, &prog , &prog_length);
	glCompileShader(shader);

	if (GL_COMPILE_STATUS == GL_FALSE) {
		clog << "\nCompilation failed !";
	}
	
	char* buffer = new char[1000];
	int length;
	
	glGetShaderInfoLog(shader, 1000, &length, buffer);
	clog << "\n" << buffer;

	delete [] buffer;
}

unsigned int Shader::getShader() {
	return shader;
}

GLenum Shader::getShaderType() {
	return shaderType;	
}
