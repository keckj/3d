#include "cylinder.h"

#include <iostream>

/* Number of points used to draw the cylinder */
int Cylinder::NB_POINTS = 100;

/* Constructor */
Cylinder::Cylinder (float radius, float height) : radius(radius), height(height) {
}

/* Drawing Method */
void Cylinder::draw () {
    // TODO: pourquoi ça marche pas quand on le met dans le constructeur ?
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

Cylinder::~Cylinder() {
    gluDeleteQuadric(cyl);
    gluDeleteQuadric(d);
}

