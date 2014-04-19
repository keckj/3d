#ifndef LEFTFOREARM_H
#define LEFTFOREARM_H

#include "Arm.h"

class LeftForeArm : public Arm {
    public:
        LeftForeArm (float width, float height);

        void animateDownwards();

    private:
        bool down;
        float theta;
};

#endif
