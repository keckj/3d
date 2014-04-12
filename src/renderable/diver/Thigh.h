#ifndef THIGH_H
#define THIGH_H

#include "RenderTree.h"
#include "Cylinder.h"

class Thigh : public RenderTree {
    public:
        Thigh (float width, float height);

        float getWidth () const;
        float getHeight () const;

    protected:
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);

    private:
        Cylinder cyl;
};

#endif

