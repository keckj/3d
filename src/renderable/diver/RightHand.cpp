#include "RightHand.h"

RightHand::RightHand () : Hand() {
}

void RightHand::drawDownwards(const float *currentTransformationMatrix) {
    Hand::drawDownwards(currentTransformationMatrix);
}

