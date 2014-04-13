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
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
    cyl.draw();

    glPopMatrix();
}

