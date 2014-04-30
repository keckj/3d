#ifndef JOINT_H
#define JOINT_H

#include "renderable.h"
#include "headers.h"

#include "renderTree.h"

class Joint : public RenderTree {
    public:
        Joint (float radius);
        float getRadius () const;

    protected:
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        void drawUpwards(const float *currentTransformationMatrix = consts::identity4);

    private:
        float radius;
};

#endif

