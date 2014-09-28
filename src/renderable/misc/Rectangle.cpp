#include "Rectangle.h"

Rectangle::Rectangle (float width, float height, float depth) : width(width), height(height), depth(depth), vertices(NULL) {
}

float Rectangle::getWidth () const {
    return width;
}

float Rectangle::getHeight () const {
    return height;
}

float Rectangle::getDepth () const {
    return depth;
}

void Rectangle::draw () {
    GLfloat s0[] = {+depth / 2, -width / 2, -height / 2};
    GLfloat s1[] = {+depth / 2, +width / 2, -height / 2};
    GLfloat s2[] = {-depth / 2, +width / 2, -height / 2};
    GLfloat s3[] = {-depth / 2, -width / 2, -height / 2};
    GLfloat s4[] = {+depth / 2, -width / 2, +height / 2};
    GLfloat s5[] = {+depth / 2, +width / 2, +height / 2};
    GLfloat s6[] = {-depth / 2, +width / 2, +height / 2};
    GLfloat s7[] = {-depth / 2, -width / 2, +height / 2};

    glBegin(GL_QUADS);
        glNormal3f(0.0, 0.0, -1.0); // same normal shared by 4 vertices
        glVertex3fv(s0);   // direct coordinates
        glVertex3fv(s1);                // stored coordinates
        glVertex3fv(s2);
        glVertex3fv(s3);

        glNormal3f(0.0, -1.0, 0.0);
        glVertex3fv(s0);
        glVertex3fv(s4);
        glVertex3fv(s7);
        glVertex3fv(s3);

        // 1 5 4 0
        glNormal3f(1.0, 0.0, 0.0);
        glVertex3fv(s1);
        glVertex3fv(s5);
        glVertex3fv(s4);
        glVertex3fv(s0);

        // 2 6 5 1
        glNormal3f(0.0, 1.0, 0.0);
        glVertex3fv(s2);
        glVertex3fv(s6);
        glVertex3fv(s5);
        glVertex3fv(s1);

        // 3 7 6 2
        glNormal3f(-1.0, 0.0, 0.0);
        glVertex3fv(s3);
        glVertex3fv(s7);
        glVertex3fv(s6);
        glVertex3fv(s2);

        // 4 5 6 7
        glNormal3f(0.0, 0.0, 1.0);
        glVertex3fv(s4);
        glVertex3fv(s5);
        glVertex3fv(s6);
        glVertex3fv(s7);
    glEnd();
}

Rectangle::~Rectangle() {
    delete [] vertices;
}

