
#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class Shader {

	protected:
		unsigned int shader;
		GLenum shaderType;

	public:
		Shader(const char* location, GLenum shaderType);

		unsigned int getShader();
		GLenum getShaderType();
};
