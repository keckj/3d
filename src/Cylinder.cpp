#include <iostream>

#include "cylinder.h"

int Cylinder::NB_POINTS = 100;

Cylinder::Cylinder (float radius, float height) : radius(radius), height(height) {
}

void Cylinder::draw () {
    cyl = gluNewQuadric();
    d = gluNewQuadric();

    glPushMatrix();

    gluQuadricDrawStyle(cyl, GLU_FILL);
    gluCylinder(cyl, radius, radius, height, NB_POINTS, 1);

    glPushMatrix();
    glRotatef(180, 0, 1, 0);
    gluDisk(d, 0, radius, NB_POINTS, 1);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0, 0, height);
    gluDisk(d, 0, radius, NB_POINTS, 1);
    glPopMatrix();

    glPopMatrix();
}

float Cylinder::getRadius () const {
    return radius;
}

float Cylinder::getHeight () const {
    return height;
}

Cylinder::~Cylinder() {
    gluDeleteQuadric(cyl);
    gluDeleteQuadric(d);
}

