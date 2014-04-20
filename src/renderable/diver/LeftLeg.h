#ifndef LEFTLEG_H
#define LEFTLEG_H

#include "Leg.h"

class LeftLeg : public Leg {
    public:
        LeftLeg (float width, float height);

        void animateDownwards ();

    private:
        bool down;
        float theta;
};

#endif
