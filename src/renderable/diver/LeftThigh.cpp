#include "LeftThigh.h"
#include "Dimensions.h"
#include "JointLeftThigh.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

LeftThigh::LeftThigh (float width, float height) : Leg(width, height) {
    JointLeftThigh *jointLeftThigh = new JointLeftThigh(RADIUS_JOINT);
    addChild("jointLeftThigh", jointLeftThigh);
    translateChild("jointLeftThigh", 0, 0, 0);
}

void LeftThigh::drawDownwards (const float * currentTransformationMatrix) {
    glColor3ub(165, 93, 53);
    Leg::drawDownwards(currentTransformationMatrix);
    glColor3ub(255, 255, 255);
}

void LeftThigh::animateDownwards () {
    float pas = 0.05f;

    if (down) {
        rotate(Quaternion(Vec(0, 1, 0), pas));
        theta -= pas;
    } else {
        rotate(Quaternion(Vec(0, 1, 0), -pas));
        theta += pas;
    }

    if (theta > M_PI / 6) {
        down = true;
    } else if (theta < -M_PI / 6) {
        down = false;
    }
}
