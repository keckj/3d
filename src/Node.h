#ifndef NODE_H
#define NODE_H

#include "renderable.h"
#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

class Node : public Renderable {
    public:
        void draw();

    private:
};

#endif
