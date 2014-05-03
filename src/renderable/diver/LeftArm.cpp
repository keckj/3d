#include "LeftArm.h"
#include "LeftHand.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

LeftArm::LeftArm (float width, float height) : Arm(width, height) {
    LeftHand *leftHand = new LeftHand();
    addChild("leftHand", leftHand);
    translateChild("leftHand", -0.04, -getWidth() / 2, getHeight() / (4.5));
    rotateChild("leftHand", Quaternion(Vec(0, 1, 0), M_PI));
    scaleChild("leftHand", 0.8, 0.8, 0.5);
}

void LeftArm::drawDownwards(const float *currentTransformationMatrix) {
    /* glColor3ub(239, 208, 102); */
    Arm::drawDownwards(currentTransformationMatrix);
    glColor3ub(255, 255, 255);
}

void LeftArm::animateDownwards () {
    float pas = 0.05f;

    if (down) {
        rotate(Quaternion(Vec(0, 1, 0), pas));
        theta -= pas;
    } else {
        rotate(Quaternion(Vec(0, 1, 0), -pas));
        theta += pas;
    }

    if (theta > M_PI - M_PI / 6) {
        down = true;
    } else if (theta < 0) {
        down = false;
    }
}

