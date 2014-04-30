#include "RightForeArm.h"
#include "Dimensions.h"
#include "JointRightArm.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

RightForeArm::RightForeArm (float width, float height) : Arm(width, height), down(true) {
    JointRightArm *jointRightArm = new JointRightArm(RADIUS_JOINT);
    addChild("jointRightArm", jointRightArm);
    translateChild("jointRightArm", 0, 0, 0);
}

void RightForeArm::drawDownwards (const float *currentTransformationMatrix) {
    glColor3ub(165, 93, 53);
    Arm::drawDownwards(currentTransformationMatrix);
    glColor3ub(255, 255, 255);
}

void RightForeArm::animateDownwards() {
    float pas = 0.05f;

    if (down) {
        rotate(Quaternion(Vec(0, 1, 0), pas));
        theta -= pas;
    } else {
        rotate(Quaternion(Vec(0, 1, 0), -pas));
        theta += pas;
    }

    if (theta > M_PI / 4) {
        down = true;
    } else if (theta < -M_PI / 6) {
        down = false;
    }
}

