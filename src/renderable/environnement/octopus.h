#ifndef OCTOPUS_H
#define OCTOPUS_H

#include "renderTree.h"
#include "object.h"
#include "CardinalSpline.h"
#include <vector>

class Octopus : public RenderTree {
	
	public:
		Octopus(const std::vector<Object*> &objs);
		~Octopus();
		
		void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
		void animateDownwards();

	private:
};

#endif
