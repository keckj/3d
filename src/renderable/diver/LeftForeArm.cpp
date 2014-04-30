#include "LeftForeArm.h"
#include "Dimensions.h"
#include "JointLeftArm.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

LeftForeArm::LeftForeArm (float width, float height) : Arm(width, height) {
    JointLeftArm *jointLeftArm = new JointLeftArm(RADIUS_JOINT);
    addChild("jointLeftArm", jointLeftArm);
    translateChild("jointLeftArm", 0, 0, 0);
}

void LeftForeArm::drawDownwards (const float *currentTransformationMatrix) {
    glColor3ub(165, 93, 53);
    Arm::drawDownwards(currentTransformationMatrix);
    glColor3ub(255, 255, 255);
}

void LeftForeArm::animateDownwards() {
    float pas = 0.05f;

    if (down) {
        rotate(Quaternion(Vec(0, 1, 0), -pas));
        theta -= pas;
    } else {
        rotate(Quaternion(Vec(0, 1, 0), pas));
        theta += pas;
    }

    if (theta > M_PI / 4) {
        down = true;
    } else if (theta < -M_PI / 6) {
        down = false;
    }
}

