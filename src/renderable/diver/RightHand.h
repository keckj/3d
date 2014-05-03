#ifndef RIGHTHAND_H
#define RIGHTHAND_H

#include "Hand.h"

class RightHand : public Hand {
    public:
        RightHand ();

        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
};

#endif

