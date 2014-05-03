#ifndef HAND_H
#define HAND_H

#include "renderTree.h"
#include "ObjLoader.h"

class Hand : public RenderTree {
    public:
        Hand ();
        ~ Hand();

    protected:
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        void drawUpwards(const float *currentTransformationMatrix = consts::identity4);

    private:
        ObjLoader *hand;
};

#endif

