#ifndef SEADIVER_H
#define SEADIVER_H

#include "Dimensions.h"
#include "RenderTree.h"

class SeaDiver : public RenderTree {
    public:
        SeaDiver ();

        // Events
        void keyPressEvent(QKeyEvent* e);
        void mouseMoveEvent(QMouseEvent* e);

    protected:
        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);

    private:
};

#endif

