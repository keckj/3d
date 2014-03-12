#include "renderable.h"

float Renderable::Identity[] = {1.0f, 0.0f, 0.0f,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f};

float* Renderable::getModelMatrix () const {
    return Identity;
}

