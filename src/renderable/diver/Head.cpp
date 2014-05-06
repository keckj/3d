#include "globals.h"
#include "Head.h"
#include "audible.h"

Head::Head (float radius) : RenderTree(), radius(radius) {
    bubbles = new Audible("sounds/diver/water014.wav", qglviewer::Vec(0,0,0));
    bubblesPlaying = false;
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

void Head::animateDownwards() {
    if (!bubblesPlaying) {
        bubbles->playSource();
        bubblesPlaying = true;
    }
    bubbles->setSourcePosition(Globals::pos);
}
