
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <cstdio>
#include "window.hpp"

using namespace std;

Window::Window(unsigned int width, unsigned int height, const char *title) {


	window = glfwCreateWindow (width, height, title, NULL, NULL);

	if (!window) {
		clog << "ERROR: could not open window with GLFW3\n";

		glfwTerminate();
		return;
	}

	glfwMakeContextCurrent(window);

	// start GLEW extension handler
	glewExperimental = GL_TRUE;
	glewInit();

	// Get version info
	const GLubyte* renderer = glGetString (GL_RENDERER); // get renderer string
	const GLubyte* version = glGetString (GL_VERSION); // version as a string

	clog << "\nRenderer : " << renderer;
	clog << "\nOpenGL version supported : " << version;

	// tell GL to only draw onto a pixel if the shape is closer to the viewer
	glEnable (GL_DEPTH_TEST); // enable depth-testing
	glDepthFunc (GL_LESS); // depth-testing interprets a smaller value as "closer"
}


int Window::close() {
	clog << "\nClosing window";

	glfwTerminate();

	return 0;
}

GLFWwindow *Window::getWindow() {
	return window;
}
