#include "SeaDiver.h"

#include "Pipe.h"

#include "Arm.h"
#include "Trunk.h"
#include "Thigh.h"
#include "Head.h"

SeaDiver::SeaDiver() : RenderTree() {
    Trunk *trunk = new Trunk(WIDTH_TRUNK, HEIGHT_TRUNK, DEPTH_TRUNK);
    addChild("trunk", trunk);

    Thigh *leftThigh = new Thigh(WIDTH_THIGH, HEIGHT_THIGH);
    addChild("leftThigh", leftThigh);
    translateChild("leftThigh", 0, (trunk->getWidth() - leftThigh->getWidth()) / 2, -1.5 * trunk->getHeight());

    Thigh *rightThigh = new Thigh(WIDTH_THIGH, HEIGHT_THIGH);
    addChild("rightThigh", rightThigh);
    translateChild("rightThigh", 0, (-trunk->getWidth() + leftThigh->getWidth()) / 2, -1.5 * trunk->getHeight());

    Arm *leftForearm = new Arm(WIDTH_FOREARM, HEIGHT_FOREARM);
    addChild("leftForearm", leftForearm);
    rotateChild("leftForearm", qglviewer::Quaternion(Vec(1, 0, 0), M_PI / 2));
    translateChild("leftForearm", 0, trunk->getWidth() / 2, 0);

    Arm *rightForearm = new Arm(WIDTH_FOREARM, HEIGHT_FOREARM);
    addChild("rightForearm", rightForearm);
    rotateChild("rightForearm", qglviewer::Quaternion(Vec(1, 0, 0), -M_PI / 2));
    translateChild("rightForearm", 0, -trunk->getWidth() / 2, 0);

    Head *head = new Head(RADIUS_HEAD);
    addChild("head", head);
    translateChild("head", 0, 0, HEIGHT_NECK + trunk->getHeight() / 2);
}

void SeaDiver::drawDownwards(const float *currentTransformationMatrix) {
}

void SeaDiver::animate () {
    // TODO
}

// Events
void SeaDiver::keyPressEvent(QKeyEvent* e) {
    // TODO
}

void SeaDiver::mouseMoveEvent(QMouseEvent* e) {
    // TODO
}

