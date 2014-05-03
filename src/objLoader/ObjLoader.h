#ifndef OBJLOADER_H
#define OBJLOADER_H

#include <string>
#include <vector>

#include "program.h"
#include "object.h"

#include "tiny_obj_loader.h"
#include "renderTree.h"

class ObjLoader : public RenderTree {
    public:
        ObjLoader (std::string const& file, std::string const& basepath = "obj_files/");
        ~ObjLoader();
        std::vector<Object*> getObjects();
        void print();
        void drawDownwards (const float * curretnTransformationMatrix = consts::identity4);

    private:
        void makeProgram();

        std::string objFilename;
        std::string mtlFilename;
        std::vector<tinyobj::shape_t> shapes;
};

#endif

