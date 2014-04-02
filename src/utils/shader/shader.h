
#pragma once

#include <GL/glew.h>

class Shader {

	protected:
		unsigned int shader;
		GLenum shaderType;

	public:
		Shader(const char* location, GLenum shaderType);

		unsigned int getShader();
		GLenum getShaderType();
};
