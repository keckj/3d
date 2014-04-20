#ifndef RIGHTARM_H
#define RIGHTARM_H

#include "Arm.h"

class RightArm : public Arm {
    public:
        RightArm (float width, float height);

        void animateDownwards ();

    private:
        bool down;
        float theta;
};

#endif
