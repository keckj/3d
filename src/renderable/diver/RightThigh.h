#ifndef RIGHTTHIGH_H
#define RIGHTTHIGH_H

#include "Leg.h"

class RightThigh : public Leg {
    public:
        RightThigh (float width, float height);

        void animateDownwards ();

    private:
        bool down;
        float theta;
};

#endif
