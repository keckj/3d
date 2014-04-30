#ifndef JOINTRIGHTUPTRUNK_H
#define JOINTRIGHTUPTRUNK_H

#include "renderable.h"
#include "headers.h"

#include "Joint.h"
#include "RightForeArm.h"
#include "Dimensions.h"

class JointRightUpTrunk : public Joint {
    public:
        JointRightUpTrunk (float radius) : Joint(radius) {
            RightForeArm *rightForearm = new RightForeArm(WIDTH_FOREARM, HEIGHT_FOREARM);
            addChild("rightForearm", rightForearm);
            translateChild("rightForearm", 0, 0, -rightForearm->getHeight());
        };

        void drawDownwards (const float * currentTransformationMatrix) {
            glColor3ub(165, 93, 53);
            Joint::drawDownwards(currentTransformationMatrix);
            glColor3ub(255, 255, 255);
        };
};

#endif

