#include "Arm.h"

Arm::Arm(float width, float height) : RenderTree(), cyl(width / 2, height) {
}

float Arm::getWidth() const {
    return 2 * cyl.getRadius();
}

float Arm::getHeight() const {
    return cyl.getHeight();
}

void Arm::drawDownwards(const float *currentTransformationMatrix) {
    std::cout << "down arm" << std::endl;
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
    cyl.draw();
}

void Arm::drawUpwards (const float *currentTransformationMatrix) {
    std::cout << "up arm" << std::endl;
    glPopMatrix();
}

