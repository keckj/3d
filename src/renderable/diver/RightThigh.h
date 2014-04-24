#ifndef RIGHTTHIGH_H
#define RIGHTTHIGH_H

#include "Leg.h"

class RightThigh : public Leg {
    public:
        RightThigh (float width, float height);

        void drawDownwards (const float * currentTransformationMatrix);
        void animateDownwards ();

    private:
        bool down;
        float theta;
};

#endif
