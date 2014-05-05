#ifndef SEADIVER_H
#define SEADIVER_H

#include "Dimensions.h"
#include "renderTree.h"
#include "CardinalSpline.h"

class SeaDiver : public RenderTree {
    public:
        SeaDiver ();

        // Events
        void keyPressEvent(QKeyEvent* e);
        void mouseMoveEvent(QMouseEvent* e);

    protected:
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        void drawUpwards(const float *currentTransformationMatrix = consts::identity4);
        void animateDownwards();

    private:
        float t;
        float dt;
        CardinalSpline *cs;
        Vec pos;
};

#endif

