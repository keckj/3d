#ifndef OBJLOADER_H
#define OBJLOADER_H

#include <string>
#include <vector>

#include "tiny_obj_loader.h"

#include "renderable.h"
#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

class ObjLoader : public Renderable {
    public:
        ObjLoader (std::string const& file, std::string const& basepath = "obj_files/");
        void print ();
        void draw();

    private:
        std::string objFilename;
        std::string mtlFilename;
        std::vector<tinyobj::shape_t> shapes;
};

#endif

