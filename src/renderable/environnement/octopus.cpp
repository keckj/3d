#include "octopus.h"

Octopus::Octopus(const std::vector<Object*> &objs) {
    for (unsigned int i = 0; i < objs.size(); i++) {
        std::stringstream s;
        s << "octopus=" << i;
        this->addChild(s.str(),objs[i]);
    }

    scale(0.001);
    rotate(Quaternion(Vec(0, 1, 0), M_PI));
    translate(0, -23, -1);
}

Octopus::~Octopus() {
}
		
void Octopus::drawDownwards(const float *currentTransformationMatrix) {
}

void Octopus::animateDownwards() {
}

