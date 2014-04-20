#include "LeftForeArm.h"
#include "LeftArm.h"
#include "Dimensions.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

LeftForeArm::LeftForeArm (float width, float height) : Arm(width, height) {
    LeftArm *leftArm = new LeftArm(WIDTH_ARM, HEIGHT_ARM);
    addChild("leftArm", leftArm);
    translateChild("leftArm", 0, 0, -getHeight());
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

