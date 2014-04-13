#include "SeaDiver.h"

#include "Pipe.h"

#include "Arm.h"
#include "Trunk.h"
#include "Leg.h"
#include "Head.h"

SeaDiver::SeaDiver() : RenderTree() {
    Trunk *trunk = new Trunk(WIDTH_TRUNK, HEIGHT_TRUNK, DEPTH_TRUNK);
    addChild("trunk", trunk);
}

void SeaDiver::drawDownwards(const float *currentTransformationMatrix) {
}

// Events
void SeaDiver::keyPressEvent(QKeyEvent* e) {
    // TODO
}

void SeaDiver::mouseMoveEvent(QMouseEvent* e) {
    // TODO
}

