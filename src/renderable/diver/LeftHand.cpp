#include "LeftHand.h"

LeftHand::LeftHand () : Hand() {
}

void LeftHand::drawDownwards(const float *currentTransformationMatrix) {
    Hand::drawDownwards(currentTransformationMatrix);
}

