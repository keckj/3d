#include "RightArm.h"
#include "RightHand.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

RightArm::RightArm (float width, float height) : Arm(width, height) {
    RightHand *rightHand = new RightHand();
    addChild("rightHand", rightHand);
    translateChild("rightHand", -0.04, -getWidth() / 2, getHeight() / (4.5));
    rotateChild("rightHand", Quaternion(Vec(0, 1, 0), M_PI));
    rotateChild("rightHand", Quaternion(Vec(0, 0, 1), M_PI));
    scaleChild("rightHand", 0.8, 0.8, 0.5);
}

void RightArm::drawDownwards(const float *currentTransformationMatrix) {
    /* glColor3ub(239, 208, 102); */
    Arm::drawDownwards(currentTransformationMatrix);
    glColor3ub(255, 255, 255);
}

void RightArm::animateDownwards () {
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

