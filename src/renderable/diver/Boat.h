#ifndef BOAT_H
#define BOAT_H

#include "headers.h"
#include "RenderTree.h"
#include "ObjLoader.h"

class Boat : public RenderTree {
    public:
        Boat ();
        ~Boat ();

        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        void drawUpwards(const float *currentTransformationMatrix = consts::identity4);
        void animateDownwards ();

    private:
        ObjLoader *boat;

        float theta;
        bool right;
};

#endif

