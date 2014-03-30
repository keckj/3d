
#include "renderTree.hpp"
#include "log.hpp"
#include <cassert>

RenderTree::RenderTree(Renderable *object, bool active) 
: object(object), active(active)
{
}

RenderTree::~RenderTree() {
}

void RenderTree::draw(const float *currentTransformationMatrix) const {
	
	//if not active draw nothing
	if(!this->active)
		return;
	
	//draw current object
	if(this->object != 0) {
		const float *newTransformationMatrix = multMat4f(currentTransformationMatrix, this->object->getRelativeModelMatrix());
		this->object->draw(newTransformationMatrix);
	
		//draw subtrees
		std::map<std::string, RenderTree*>::const_iterator it;
		for (it = children.cbegin(); it != children.cend(); it++) {
				it->second->draw(newTransformationMatrix);
		}

		delete [] newTransformationMatrix;
	}
	//else just draw substrees with current matrix
	else {
		std::map<std::string, RenderTree*>::const_iterator it;
		for (it = children.cbegin(); it != children.cend(); it++) {
			it->second->draw(currentTransformationMatrix);
		}
	}
}

void RenderTree::addChild(std::string key, RenderTree *child) {
	assert(children.insert(std::pair<std::string,RenderTree*>(key, child)).second == true);	
}

void RenderTree::removeChild(std::string key) {
	assert(children.erase(key) == 1);
}

void RenderTree::desactivateChild(std::string childName) {
	std::map<std::string, RenderTree*>::iterator it;

	it = children.find(childName);
	assert(it != children.end());

	it->second->active = false;
}

void RenderTree::activateChild(std::string childName) {
	std::map<std::string, RenderTree*>::iterator it;

	it = children.find(childName);
	assert(it != children.end());

	it->second->active = false;
}
		
const float* RenderTree::multMat4f(const float *m1, const float* m2) {
	float *m = new float[16]();
	
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			for (int k = 0; k < 4; k++) {
				m[4*i+j] += m1[4*i+k] * m2[4*k+j];
			}
		}
	}
	
	const float *cm = {m};
	
	//log_console.infoStream() << "Transformation Matrix M1:";
	//for (int i = 0; i < 4; i++) {
		//log_console.infoStream() 
			//<< "\t" << m1[4*i+0] 
			//<< "\t" << m1[4*i+1] 
			//<< "\t" << m1[4*i+2] 
			//<< "\t" << m1[4*i+3];
	//}
	
	//log_console.infoStream() << "Transformation Matrix M2:";
	//for (int i = 0; i < 4; i++) {
		//log_console.infoStream() 
			//<< "\t" << m2[4*i+0] 
			//<< "\t" << m2[4*i+1] 
			//<< "\t" << m2[4*i+2] 
			//<< "\t" << m2[4*i+3];
	//}


	//log_console.infoStream() << "Transformation Matrix M:";
	//for (int i = 0; i < 4; i++) {
		//log_console.infoStream() 
			//<< "\t" << m[4*i+0] 
			//<< "\t" << m[4*i+1] 
			//<< "\t" << m[4*i+2] 
			//<< "\t" << m[4*i+3];
	//}
	//
	
	return cm;

}
