#include "RightForeArm.h"
#include "RightArm.h"
#include "Dimensions.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

RightForeArm::RightForeArm (float width, float height) : Arm(width, height), down(true) {
    RightArm *rightArm = new RightArm(WIDTH_ARM, HEIGHT_ARM);
    addChild("rightArm", rightArm);
    translateChild("rightArm", 0, 0, -getHeight());
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

