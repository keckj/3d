#pragma once

#include <iostream>
using namespace std;

void log_gl_params () {

	GLenum params[] = {
		GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS,
		GL_MAX_CUBE_MAP_TEXTURE_SIZE,
		GL_MAX_DRAW_BUFFERS,
		GL_MAX_FRAGMENT_UNIFORM_COMPONENTS,
		GL_MAX_TEXTURE_IMAGE_UNITS,
		GL_MAX_TEXTURE_SIZE,
		GL_MAX_VARYING_FLOATS,
		GL_MAX_VERTEX_ATTRIBS,
		GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS,
		GL_MAX_VERTEX_UNIFORM_COMPONENTS,
		GL_MAX_VIEWPORT_DIMS,
		GL_STEREO,
	};
	const char* names[] = {
		"GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS",
		"GL_MAX_CUBE_MAP_TEXTURE_SIZE",
		"GL_MAX_DRAW_BUFFERS",
		"GL_MAX_FRAGMENT_UNIFORM_COMPONENTS",
		"GL_MAX_TEXTURE_IMAGE_UNITS",
		"GL_MAX_TEXTURE_SIZE",
		"GL_MAX_VARYING_FLOATS",
		"GL_MAX_VERTEX_ATTRIBS",
		"GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS",
		"GL_MAX_VERTEX_UNIFORM_COMPONENTS",
		"GL_MAX_VIEWPORT_DIMS",
		"GL_STEREO",
	};

	clog << "\n\nGL Context Params:";
	char msg[256];
	// integers - only works if the order is 0-10 integer return types
	for (int i = 0; i < 10; i++) {
		int v = 0;
		glGetIntegerv (params[i], &v);
		clog << "\n" << names[i] << " " << v;
	}
	
	// others
	int v[2];
	v[0] = v[1] = 0;

	glGetIntegerv (params[10], v);
	sprintf (msg, "%s %i %i", names[10], v[0], v[1]);
	clog << msg;

	unsigned char s = 0;
	glGetBooleanv (params[11], &s);
	sprintf (msg, "%s %i", names[11], (unsigned int)s);
	clog << "\n" << msg << "\n";
}

