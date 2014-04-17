#include "RightThigh.h"
#include "RightLeg.h"
#include "Dimensions.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

RightThigh::RightThigh (float width, float height) : Leg(width, height) {
    RightLeg *rightLeg = new RightLeg(WIDTH_LEG, HEIGHT_LEG);
    addChild("rightLeg", rightLeg);
    translateChild("rightLeg", 0, 0, -getHeight());
}

void RightThigh::animateDownwards () {
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
    } else if (theta < -M_PI / 4) {
        down = false;
    }
}
