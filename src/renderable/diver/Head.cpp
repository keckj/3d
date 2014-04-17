#include "Head.h"

Head::Head (float radius) : RenderTree(), radius(radius) {
}

float Head::getRadius () const {
    return radius;
}

void Head::drawDownwards(const float *currentTransformationMatrix) {
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
    glutSolidSphere(radius, 20, 20);
}

void Head::drawUpwards (const float *currentTransformationMatrix) {
    glPopMatrix();
}
