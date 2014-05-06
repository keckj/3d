#include "Boat.h"

#include <cmath>
#include <QGLViewer/vec.h>
using namespace qglviewer;

#define THETAMAX (0.1)

Boat::Boat(const std::vector<Object*> &objs) {
	this->addChild("boat0",objs[0]);
	this->addChild("boat1",objs[1]);
	this->addChild("boat2",objs[2]);
	this->addChild("boat3",objs[3]);
	this->addChild("boat4",objs[4]);
	this->addChild("boat5",objs[5]);
	this->addChild("boat6",objs[6]);
	this->addChild("boat7",objs[7]);
	this->addChild("boat8",objs[8]);
	this->addChild("boat9",objs[9]);
	this->addChild("boat10",objs[10]);
	this->addChild("boat11",objs[11]);

    scale(0.01);
    translate(0, 11, 0);
}

void Boat::drawDownwards(const float *currentTransformationMatrix) {
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

