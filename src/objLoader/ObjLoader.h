#ifndef OBJLOADER_H
#define OBJLOADER_H

#include <string>
#include <vector>

#include "renderable.h"
#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

class ObjLoader : public Renderable {
    public:
        ObjLoader (std::string const& filename);
        void print ();
        void draw();
        ~ ObjLoader ();

    private:
        std::string filename;
        GLfloat* vector2float (std::vector<float>& array);
        std::vector<std::string> splitOnWS (std::string const& str);
        int countSlashes (std::string const& str);
        void strToVector (std::string const& str, float &x, float &y, float &z);
        void parse ();

        GLfloat* vertices;
        GLfloat* normals;
        GLfloat* textures;
        unsigned int nv, nn;
};

#endif

