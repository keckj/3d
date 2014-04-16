#ifndef TRUNK_H
#define TRUNK_H

#include "renderTree.h"
#include "Rectangle.h"

class Trunk : public RenderTree {
    public:
        Trunk (float width, float height, float depth);

        float getWidth () const;
        float getHeight() const;
        float getDepth () const;

    protected:
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);

    private:
        Rectangle rect;
};

#endif

