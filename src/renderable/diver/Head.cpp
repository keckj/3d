#include "Head.h"

Head::Head (float radius) : BodyPart(true), radius(radius) {
}

float Head::getRadius () const {
    return radius;
}

void Head::draw () {
    glPushMatrix();

    glutSolidSphere(radius, 20, 20);

    glPopMatrix();
}

