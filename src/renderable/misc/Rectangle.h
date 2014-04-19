#ifndef RECTANGLE_H
#define RECTANGLE_H

#include "renderable.h"
#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

class Rectangle : public Renderable {
    public:
        Rectangle (float width, float height, float depth);
        ~Rectangle();

        float getWidth () const;
        float getHeight () const;
        float getDepth () const;

        void draw ();

    private:
        float width;
        float height;
        float depth;

        GLfloat *vertices;
};

#endif

