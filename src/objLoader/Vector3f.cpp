#include "Vector3f.h"

Vector3f::Vector3f (float x, float y, float z) : x(x), y(y), z(z) {
}

float Vector3f::getX () const {
    return x;
}

float Vector3f::getY () const {
    return y;
}

float Vector3f::getZ () const {
    return z;
}

