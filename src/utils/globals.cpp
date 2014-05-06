
#include "headers.h"
#include "globals.h"
#include "utils.h"
#include "log.h"

int Globals::glMax3DTextureSize = 0;
int Globals::glMaxTextureSize = 0;
int Globals::glMaxVertexAttribs = 0;
int Globals::glMaxDrawBuffers = 0;
int Globals::glMaxCombinedTextureImageUnits = 0;
int Globals::glMaxVertexUniformBlocks = 0;
int Globals::glMaxGeometryUniformBlocks = 0;
int Globals::glMaxFragmentUniformBlocks = 0;
int Globals::glMaxUniformBlockSize = 0;

float *Globals::glPointSizeRange = 0;
float Globals::glPointSizeGranularity = 0;
float Globals::glPointSize = 0;

const unsigned char *Globals::glVersion = 0;
const unsigned char *Globals::glShadingLanguageVersion = 0;

Viewer *Globals::viewer = 0;
unsigned int Globals::projectionViewUniformBlock = 0;

void Globals::init() {

        glVersion = glGetString(GL_VERSION);
        glShadingLanguageVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);

        glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &glMaxVertexAttribs);
        glGetIntegerv(GL_MAX_DRAW_BUFFERS, &glMaxDrawBuffers);
        glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &glMaxCombinedTextureImageUnits);
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &glMaxTextureSize);
        glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &glMax3DTextureSize);
        glGetIntegerv(GL_MAX_VERTEX_UNIFORM_BLOCKS, &glMaxVertexUniformBlocks);
        glGetIntegerv(GL_MAX_GEOMETRY_UNIFORM_BLOCKS, &glMaxGeometryUniformBlocks);
        glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_BLOCKS, &glMaxFragmentUniformBlocks);
        glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &glMaxUniformBlockSize);

        glPointSizeRange = new float[2];
        glGetFloatv(GL_POINT_SIZE_RANGE, glPointSizeRange);

        glGetFloatv(GL_POINT_SIZE_GRANULARITY, &glPointSizeGranularity);
        glGetFloatv(GL_POINT_SIZE, &glPointSize);

        glGenBuffers(1, &projectionViewUniformBlock);
        glBindBuffer(GL_UNIFORM_BUFFER, Globals::projectionViewUniformBlock);
        glBufferData(GL_UNIFORM_BUFFER, (16+16+4+4+4+4)*sizeof(GLfloat), 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        log_console.infoStream() << "[Global Vars Init]";
}

void Globals::check() {
        assert(projectionViewUniformBlock != 0);
}

void Globals::print(std::ostream &out) {
        out << "[Globals]";
        out << "\n\tGL_VERSION " << glVersion;
        out << "\n\tGL_SHADING_LANGUAGE_VERSION " << glShadingLanguageVersion;
        out << "\n\tGL_MAX_TEXTURE_SIZE " << glMaxTextureSize << " x " << glMaxTextureSize;
        out << "\n\tGL_MAX_3D_TEXTURE_SIZE " << glMax3DTextureSize << " x " << glMax3DTextureSize << " x " << glMax3DTextureSize ;
        out << "\n\tGL_MAX_VERTEX_ATTRIBS " << glMaxVertexAttribs;
        out << "\n\tGL_MAX_DRAW_BUFFERS " << glMaxDrawBuffers;
        out << "\n\tGL_MAX_COMBINED_TEXTURE_IMAGE_UNITS " << glMaxCombinedTextureImageUnits;
        out << "\n\tGL_POINT_SIZE_RANGE [" << glPointSizeRange[0] << ", " << glPointSizeRange[1] << "]";
        out << "\n\tGL_POINT_SIZE_GRANULARITY " << glPointSizeGranularity;
        out << "\n\tGL_MAX_VERTEX_UNIFORM_BLOCKS " << glMaxVertexUniformBlocks;
        out << "\n\tGL_MAX_GEOMETRY_UNIFORM_BLOCKS " << glMaxGeometryUniformBlocks;
        out << "\n\tGL_MAX_FRAGMENT_UNIFORM_BLOCKS " << glMaxFragmentUniformBlocks;
        out << "\n\tGL_MAX_UNIFORM_BLOCKSIZE " << Utils::toStringMemory(glMaxUniformBlockSize);
        out << "\n";
}


