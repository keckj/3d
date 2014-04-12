#include "Thigh.h"

Thigh::Thigh(float width, float height) : RenderTree(), cyl(width / 2, height) {
}

float Thigh::getWidth() const {
    return 2 * cyl.getRadius();
}

float Thigh::getHeight() const {
    return cyl.getHeight();
}

void Thigh::drawDownwards(const float *currentTransformationMatrix) {
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
    cyl.draw();

    glPopMatrix();
}

