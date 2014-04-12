#include "Trunk.h"

#include <iostream>

Trunk::Trunk (float width, float height, float depth) : RenderTree(), rect(width, height, depth) {
}
void Trunk::drawDownwards(const float *currentTransformationMatrix) {
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
    rect.draw();

    glPopMatrix();
}

float Trunk::getWidth () const {
    return rect.getWidth();
}

float Trunk::getHeight () const {
    return rect.getHeight();
}

float Trunk::getDepth () const {
    return rect.getDepth();
}

