#include "RightThigh.h"
#include "Dimensions.h"
#include "JointRightThigh.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

RightThigh::RightThigh (float width, float height) : Leg(width, height) {
    JointRightThigh *jointRightThigh = new JointRightThigh(RADIUS_JOINT);
    addChild("jointRightThigh", jointRightThigh);
    translateChild("jointRightThigh", 0, 0, 0);
}

void RightThigh::drawDownwards (const float * currentTransformationMatrix) {
    glColor3ub(165, 93, 53);
    Leg::drawDownwards(currentTransformationMatrix);
    glColor3ub(255, 255, 255);
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
