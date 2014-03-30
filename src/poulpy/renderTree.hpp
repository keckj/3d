
#pragma once 

#include "renderable.hpp"
#include "consts.hpp"
#include <string>
#include <map>

class RenderTree {
	
	public:
	
		RenderTree(Renderable *object = 0, bool active = true);
		~RenderTree();

		void draw(const float *currentTransformationMatrix = consts::identity) const;		
		
		void addChild(std::string key, RenderTree *child);
		void removeChild(std::string key);

		void desactivateChild(std::string childName);
		void activateChild(std::string childName);


	private:
		Renderable *object;
		bool active;
		std::map<std::string, RenderTree*> children;

		static const float *multMat4f(const float *m1, const float* m2);


};
