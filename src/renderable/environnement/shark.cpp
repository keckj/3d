

#include "shark.h"

Shark::Shark(const std::vector<Object*> &objs) {
        
        this->scale(0.0002);
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


	quat.push_back(qglviewer::Quaternion(qglviewer::Vec(0,1,0),0.1).normalized());
	quat.push_back(qglviewer::Quaternion(0,1,0,0).normalized());
	quat.push_back(qglviewer::Quaternion(0,1,0,3.14).normalized());
	quat.push_back(qglviewer::Quaternion(0,1,0,0).normalized());
	quat.push_back(qglviewer::Quaternion(0,1,0,-3.14).normalized());
	quat.push_back(qglviewer::Quaternion(0,1,0,0).normalized());
	quat.push_back(qglviewer::Quaternion(0,1,0,3.14).normalized());
}

Shark::~Shark() {
	delete trajectory;
}
		
void Shark::drawDownwards(const float *currentTransformationMatrix) {
}

void Shark::animateDownwards() {

	static float t = 0.0f;
	static int n = 0;
        
        if(t>0.01)
                this->rotate(qglviewer::Quaternion::slerp(quat[n],quat[n+1],t-0.01).inverse());
        else
                this->rotate(quat[n].inverse());

        this->rotate(qglviewer::Quaternion::slerp(quat[n],quat[n+1],t,true));
        this->move((*trajectory)(n,t));

	t += 0.01;
	if(t >= 1.0f) {
		t = 0.0f;
		n++;
		n %= (points.size()-1);
	}
}
