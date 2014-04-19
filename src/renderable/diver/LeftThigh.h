#ifndef LEFTTHIGH_H
#define LEFTTHIGH_H

#include "Leg.h"

class LeftThigh : public Leg {
    public:
        LeftThigh (float width, float height);

        void animateDownwards ();

    private:
        bool down;
        float theta;
};

#endif