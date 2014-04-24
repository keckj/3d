#include "RightArm.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

RightArm::RightArm (float width, float height) : Arm(width, height) {
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

