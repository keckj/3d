#pragma once

#include <GL/glew.h>
#define GLFW_DLL
#include <GLFW/glfw3.h>

class Window {

	private:
		GLFWwindow* window;
		
	public:
		Window(unsigned int width, unsigned int height, const char *title);
		GLFWwindow *getWindow();
		int close();
};
