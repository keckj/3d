#ifndef LEFTARM_H
#define LEFTARM_H

#include "Arm.h"

class LeftArm : public Arm {
    public:
        LeftArm (float width, float height);

        void drawDownwards(const float *currentTransformationMatrix);
        void animateDownwards ();

    private:
        bool down;
        float theta;
};

#endif
