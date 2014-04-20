#include "RightLeg.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

RightLeg::RightLeg (float width, float height) : Leg(width, height) {
}

void RightLeg::animateDownwards () {
    float pas = 0.05f;

    if (down) {
        rotate(Quaternion(Vec(0, 1, 0), pas));
        theta -= pas;
    } else {
        rotate(Quaternion(Vec(0, 1, 0), -pas));
        theta += pas;
    }

    if (theta > 0) {
        down = true;
    } else if (theta < -M_PI / 2) {
        down = false;
    }
}
