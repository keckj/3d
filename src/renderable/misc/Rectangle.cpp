#include "Rectangle.h"

Rectangle::Rectangle (float width, float height, float depth) : width(width), height(height), depth(depth), vertices(NULL) {
    vertices = new GLfloat[3 * 8];

    GLfloat tmp[] = {
        +depth / 2, -width / 2, -height / 2,  // 0
        +depth / 2, +width / 2, -height / 2,  // 1
        -depth / 2, +width / 2, -height / 2,  // 2
        -depth / 2, -width / 2, -height / 2,  // 3
        +depth / 2, -width / 2, +height / 2,  // 4
        +depth / 2, +width / 2, +height / 2,  // 5
        -depth / 2, +width / 2, +height / 2,  // 6
        -depth / 2, -width / 2, +height / 2,  // 7
    };

    for (int i = 0; i < 3 * 8; i++) {
        vertices[i] = tmp[i];
    }
}

static GLubyte indices[6][4] = {
    {0, 3, 2, 1},
    {0, 4, 7, 3},
    {1, 5, 4, 0},
    {2, 6, 5, 1},
    {3, 7, 6, 2},
    {4, 5, 6, 7}
};

static GLfloat normals[6][3] = {
    { 0.0,  0.0, -1.0},
    { 0.0, -1.0,  0.0},
    { 1.0,  0.0,  0.0},
    { 0.0,  1.0,  0.0},
    {-1.0,  0.0,  0.0},
    { 0.0,  0.0,  1.0}
};

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
    glPushMatrix();

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0 , vertices);

    glNormal3fv(normals[0]);
    glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[0]);
    glNormal3fv(normals[1]);
    glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[1]);
    glNormal3fv(normals[2]);
    glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[2]);
    glNormal3fv(normals[3]);
    glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[3]);
    glNormal3fv(normals[4]);
    glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[4]);
    glNormal3fv(normals[5]);
    glDrawElements(GL_QUADS, 4, GL_UNSIGNED_BYTE, indices[5]);

    glDisableClientState(GL_VERTEX_ARRAY);

    glPopMatrix();
}

Rectangle::~Rectangle() {
    delete [] vertices;
}

