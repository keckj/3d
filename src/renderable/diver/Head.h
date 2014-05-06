#ifndef HEAD_H
#define HEAD_H

#include "globals.h"
#include "renderable.h"
#include "headers.h"
#include "audible.h"

#include "renderTree.h"

class Head : public RenderTree {
    public:
        Head (float radius);
        float getRadius () const;

    protected:
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        void drawUpwards(const float *currentTransformationMatrix = consts::identity4);
        void animateDownwards();

    private:
        float radius;

        Audible *bubbles;
        bool bubblesPlaying;
};

#endif
