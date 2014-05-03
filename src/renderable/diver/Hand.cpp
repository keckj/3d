#include "Hand.h"

Hand::Hand () {
    hand = new ObjLoader("obj_files/Hand");
}

Hand::~Hand () {
    delete hand;
}

void Hand::drawDownwards(const float *currentTransformationMatrix) {
    glPushMatrix();
    glMultTransposeMatrixf(relativeModelMatrix);

    hand->drawDownwards(currentTransformationMatrix);
}

void Hand::drawUpwards (const float *currentTransformationMatrix) {
    glPopMatrix();
}
