#ifndef ARM_H
#define ARM_H

#include "renderTree.h"
#include "Cylinder.h"

class Arm : public RenderTree {
    public:
        Arm (float width, float height);

        float getWidth () const;
        float getHeight () const;

    protected:
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        void drawUpwards(const float *currentTransformationMatrix = consts::identity4);

    private:
        Cylinder cyl;
};

#endif

