#include "Joint.h"

Joint::Joint (float radius) : RenderTree(), radius(radius) {
}

float Joint::getRadius () const {
    return radius;
}

void Joint::drawDownwards(const float *currentTransformationMatrix) {
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
    glutSolidSphere(radius, 20, 20);
    glColor3ub(255, 255, 255);
}

void Joint::drawUpwards (const float *currentTransformationMatrix) {
    glPopMatrix();
}

