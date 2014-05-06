#include "Boat.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

#define THETAMAX (0.1)

Boat::Boat () {
    boat = new ObjLoader("obj_files/boat/boat", "obj_files/boat/");
    scale(0.01);
    rotate(Quaternion(Vec(1, 0, 0), -M_PI / 2));
    translate(0, 0, 10);
}

void Boat::drawDownwards(const float *currentTransformationMatrix) {
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
    boat->drawDownwards(currentTransformationMatrix);
}

void Boat::drawUpwards(const float *currentTransformationMatrix) {
    glPopMatrix();
}

void Boat::animateDownwards () {
    if (right) {
        rotate(Quaternion(Vec(1, 0, 0), 0.01));
        theta += 0.01;
    } else {
        rotate(Quaternion(Vec(1, 0, 0), -0.01));
        theta -= 0.01;
    }

    if (theta > THETAMAX) {
        right = false;
    } else if (theta < -THETAMAX) {
        right = true;
    }
}

Boat::~Boat () {
    delete boat;
}

