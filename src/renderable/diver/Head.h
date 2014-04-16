#ifndef HEAD_H
#define HEAD_H

#include "renderable.h"
#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

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
