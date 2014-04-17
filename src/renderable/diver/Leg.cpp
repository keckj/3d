#include "Leg.h"

Leg::Leg(float width, float height) : RenderTree(), cyl(width / 2, height) {
}

float Leg::getWidth() const {
    return 2 * cyl.getRadius();
}

float Leg::getHeight() const {
    return cyl.getHeight();
}

void Leg::drawDownwards(const float *currentTransformationMatrix) {
    std::cout << "down leg" << std::endl;
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
    cyl.draw();
}

void Leg::drawUpwards (const float *currentTransformationMatrix) {
    std::cout << "up leg" << std::endl;
    glPopMatrix();
}

