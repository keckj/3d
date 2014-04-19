#ifndef HEAD_H
#define HEAD_H

#include "renderable.h"
#include "headers.h"

#include "renderTree.h"

class Head : public RenderTree {
    public:
        Head (float radius);
        float getRadius () const;

    protected:
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        void drawUpwards(const float *currentTransformationMatrix = consts::identity4);

    private:
        float radius;
};

#endif
