
#include "globals.h"
#include "log.h"
#include <GL/glew.h>

int Globals::glMaxVertexAttribs = 0;
int Globals::glMaxDrawBuffers = 0;
int Globals::glMaxCombinedTextureImageUnits = 0;

float *Globals::glPointSizeRange = 0;
float Globals::glPointSizeGranularity = 0;
float Globals::glPointSize = 0;

const unsigned char *Globals::glVersion = 0;
const unsigned char *Globals::glShadingLanguageVersion = 0;

void Globals::init() {


    glVersion = glGetString(GL_VERSION);
    glShadingLanguageVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

	glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &glMaxVertexAttribs);
	glGetIntegerv(GL_MAX_DRAW_BUFFERS, &glMaxDrawBuffers);
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &glMaxCombinedTextureImageUnits);

	glPointSizeRange = new float[2];
	glGetFloatv(GL_POINT_SIZE_RANGE, glPointSizeRange);
   
	glGetFloatv(GL_POINT_SIZE_GRANULARITY, &glPointSizeGranularity);
    glGetFloatv(GL_POINT_SIZE, &glPointSize);

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
	out << "\n\tGL_POINT_SIZE_RANGE [" << glPointSizeRange[0] << ", " << glPointSizeRange[1] << "]";
	out << "\n\tGL_POINT_SIZE_GRANULARITY " << glPointSizeGranularity;
	out << "\n\tGL_POINT_SIZE " << glPointSize;
	out << "\n";
}


