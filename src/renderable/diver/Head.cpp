#include "Head.h"

Head::Head (float radius) : RenderTree(), radius(radius) {
}

float Head::getRadius () const {
    return radius;
}

void Head::drawDownwards(const float *currentTransformationMatrix) {
    std::cout << "down head" << std::endl;
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
    glutSolidSphere(radius, 20, 20);
}

void Head::drawUpwards (const float *currentTransformationMatrix) {
    std::cout << "up head" << std::endl;
    glPopMatrix();
}
