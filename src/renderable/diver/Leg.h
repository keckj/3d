#ifndef LEG_H
#define LEG_H

#include "renderTree.h"
#include "Cylinder.h"

class Leg : public RenderTree {
    public:
        Leg (float width, float height);

        float getWidth () const;
        float getHeight () const;

    protected:
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);

    private:
        Cylinder cyl;
};

#endif

