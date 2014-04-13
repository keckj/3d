#include "RightForeArm.h"

#include <cmath>
#include <qglviewer/vec.h>
using namespace qglviewer;

RightForeArm::RightForeArm (float width, float height) : Arm(width, height), down(true) {
}

void RightForeArm::animateDownwards() {
    float pas = 0.05f;

    if (down) {
        rotate(Quaternion(Vec(1, 0, 0), -pas));
        theta -= pas;
    } else {
        rotate(Quaternion(Vec(1, 0, 0), pas));
        theta += pas;
    }

    if (theta > M_PI / 4) {
        down = true;
    } else if (theta < -M_PI / 4) {
        down = false;
    }
}

