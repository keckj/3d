
#ifndef GLOBALS_H
#define GLOBALS_H

#include "headers.h"
#include "viewer.h"
#include <ostream>
#include <string>

class Globals {

	public:
		static void init();
		static void check();
		static void print(std::ostream &out);
	
		static const unsigned char *glVersion;
		static const unsigned char *glShadingLanguageVersion;
			
		static int glMaxVertexAttribs;
		static int glMaxDrawBuffers;
		static int glMaxCombinedTextureImageUnits;

		static Viewer *viewer;
};
	
#endif /* end of include guard: GLOBALS_H */
