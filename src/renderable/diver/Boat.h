#ifndef BOAT_H
#define BOAT_H

#include "headers.h"
#include "renderTree.h"
#include "ObjLoader.h"
#include "object.h"
#include <vector>

class Boat : public RenderTree {
    public:
        Boat(const std::vector<Object*> &objs);
        ~Boat ();

        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        void animateDownwards ();

    private:
        ObjLoader *boat;

        float theta;
        bool right;
};

#endif

