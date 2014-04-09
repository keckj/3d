
#include "globals.h"
#include "log.h"
#include <GL/glew.h>

int Globals::glMaxVertexAttribs = 0;
int Globals::glMaxDrawBuffers = 0;
int Globals::glMaxCombinedTextureImageUnits = 0;

const unsigned char *Globals::glVersion = 0;
const unsigned char *Globals::glShadingLanguageVersion = 0;

void Globals::init() {

	int buffer;

    glVersion = glGetString(GL_VERSION);
    glShadingLanguageVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

	glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &buffer);
	glMaxVertexAttribs = buffer;
	
	glGetIntegerv(GL_MAX_DRAW_BUFFERS, &buffer);
	glMaxDrawBuffers = buffer;

	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &buffer);
	glMaxCombinedTextureImageUnits = buffer;

	log_console.infoStream() << "[Global Vars Init]";
}

void Globals::check() {
}

void Globals::print(std::ostream &out) {
	out << "[Globals]";
	out << "\n\tGL_VERSION " << glVersion;
	out << "\n\tGL_SHADING_LANGUAGE_VERSION " << glShadingLanguageVersion;
	out << "\n\tGL_MAX_VERTEX_ATTRIBS " << glMaxVertexAttribs;
	out << "\n\tGL_MAX_DRAW_BUFFERS " << glMaxDrawBuffers;
	out << "\n\tGL_MAX_COMBINED_TEXTURE_IMAGE_UNITS " << glMaxCombinedTextureImageUnits;
	out << "\n";
}


