

#include "shark.h"

Shark::Shark(const std::vector<Object*> &objs) {
	this->addChild("shark0",objs[0]);
	this->addChild("shark1",objs[1]);
	
	std::vector<qglviewer::Vec> points;
	points.push_back(qglviewer::Vec(0,0,0));
	points.push_back(qglviewer::Vec(0,10,0));
	points.push_back(qglviewer::Vec(0,10,10));
	points.push_back(qglviewer::Vec(10,10,10));
	trajectory = new CardinalSpline(points);
}

Shark::~Shark() {
	delete trajectory;
}
		
void Shark::drawDownwards(const float *currentTransformationMatrix) {
}

void Shark::animateDownwards() {
	this->translate(10,10,10);
}
