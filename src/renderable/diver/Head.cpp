#include "Head.h"
#include "audible.h"

Head::Head (float radius) : RenderTree(), radius(radius)/*,
    bubbles("sound/diver/water014.wav", qglviewer::Vec(0,0,0))*/ {
}

float Head::getRadius () const {
    return radius;
}

void Head::drawDownwards(const float *currentTransformationMatrix) {
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
    glutSolidSphere(radius, 20, 20);
    glColor3ub(255, 255, 255);
}

void Head::drawUpwards (const float *currentTransformationMatrix) {
    glPopMatrix();
}

/*void Head::animateDownwards() {
    bubbles.setSourcePosition(
}*/
