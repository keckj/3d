#ifndef NODE_H
#define NODE_H

#include "renderable.h"

#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

#include <string>
#include <vector>

class Node : public Renderable {
    public:
        Node (std::string const& name);
        ~ Node();

        void addSon (Node& son);
        void draw();
        void print () const;

        void setVertices (GLfloat* vertices);
        void setNormals (GLfloat* normals);
        void setTextures (GLfloat* textures);

        void setSizeVertices (unsigned int nv);
        void setSizeNormals (unsigned int nn);
        void setSizeTextures (unsigned int nt);

        void setName (std::string const& name);

    private:
        std::string name;
        std::vector<Node> sons;

        GLfloat *vertices;
        GLfloat *normals;
        GLfloat *textures;

        unsigned int nv, nn, nt;
};

#endif

