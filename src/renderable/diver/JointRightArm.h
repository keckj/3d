#ifndef JOINTRIGHTARM_H
#define JOINTRIGHTARM_H

#include "renderable.h"
#include "headers.h"

#include "Joint.h"
#include "RightArm.h"
#include "Dimensions.h"

class JointRightArm : public Joint {
    public:
        JointRightArm (float radius) : Joint(radius) {
            RightArm *rightArm = new RightArm(WIDTH_ARM, HEIGHT_ARM);
            addChild("rightArm", rightArm);
            translateChild("rightArm", 0, 0, -rightArm->getHeight());
        };

        void drawDownwards (const float * currentTransformationMatrix) {
            glColor3ub(165, 93, 53);
            Joint::drawDownwards(currentTransformationMatrix);
            glColor3ub(255, 255, 255);
        };
};

#endif

