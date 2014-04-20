#ifndef LEFTARM_H
#define LEFTARM_H

#include "Arm.h"

class LeftArm : public Arm {
    public:
        LeftArm (float width, float height);

        void animateDownwards ();

    private:
        bool down;
        float theta;
};

#endif
