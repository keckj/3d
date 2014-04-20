#ifndef RIGHTARM_H
#define RIGHTARM_H

#include "Arm.h"

class RightArm : public Arm {
    public:
        RightArm (float width, float height);

        void drawDownwards(const float *currentTransformationMatrix);
        void animateDownwards ();

    private:
        bool down;
        float theta;
};

#endif
