#include "LeftThigh.h"
#include "LeftLeg.h"
#include "Dimensions.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

LeftThigh::LeftThigh (float width, float height) : Leg(width, height) {
    LeftLeg *leftLeg = new LeftLeg(WIDTH_LEG, HEIGHT_LEG);
    addChild("leftLeg", leftLeg);
    translateChild("leftLeg", 0, 0, -getHeight());
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
