

#include "shark.h"

Shark::Shark(const std::vector<Object*> &objs) {
	this->addChild("shark0",objs[0]);
	this->addChild("shark1",objs[1]);
	
	points.push_back(qglviewer::Vec(0,0,0));
	points.push_back(qglviewer::Vec(0,10,0));
	points.push_back(qglviewer::Vec(0,10,10));
	points.push_back(qglviewer::Vec(10,10,10));
	points.push_back(qglviewer::Vec(10,0,10));
	points.push_back(qglviewer::Vec(10,0,0));
	points.push_back(qglviewer::Vec(0,0,0));
	trajectory = new CardinalSpline(points);


	quat.push_back(qglviewer::Quaternion(0,1,0,3.14));
	quat.push_back(qglviewer::Quaternion(0,1,0,0));
	quat.push_back(qglviewer::Quaternion(0,1,0,3.14));
	quat.push_back(qglviewer::Quaternion(0,1,0,0));
	quat.push_back(qglviewer::Quaternion(0,1,0,3.14));
	quat.push_back(qglviewer::Quaternion(0,1,0,0));
	quat.push_back(qglviewer::Quaternion(0,1,0,3.14));
}

Shark::~Shark() {
	delete trajectory;
}
		
void Shark::drawDownwards(const float *currentTransformationMatrix) {
}

void Shark::animateDownwards() {

	static float t = 0.0f;
	static int n = 0;

	//this->move((*trajectory)(n,t));
	//this->move(0,0,0);
	this->orientate(qglviewer::Quaternion::slerp(quat[n], quat[n+1], t, true));
	this->scale(0.002);
	
	t += 0.01;
	if(t >= 1.0f) {
		t = 0.0f;
		n++;
		n %= (points.size()-1);
	}
}
