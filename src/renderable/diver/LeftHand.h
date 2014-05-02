#ifndef LEFTHAND_H
#define LEFTHAND_H

#include "Hand.h"

class LeftHand : public Hand {
    public:
        LeftHand ();

        void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
};

#endif

