
#include "renderTree.h"
#include "object.h"
#include "CardinalSpline.h"
#include <vector>
        
class Shark : public RenderTree {
	
	public:
		Shark(const std::vector<Object*> &objs);
		~Shark();
		
		void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
		void animateDownwards();

	private:
		CardinalSpline *trajectory;
		
		std::vector<qglviewer::Vec> points;
		std::vector<qglviewer::Quaternion> quat;
};
