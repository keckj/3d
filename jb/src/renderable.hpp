
#pragma once

#include <GL/glew.h>
#include "consts.hpp"

class Renderable {

	public:
		virtual ~Renderable() {};

		virtual void draw(const float *modelMatrix = consts::identity) const = 0;
		
		virtual const float* getRelativeModelMatrix() const {
			return consts::identity;
		}
};
