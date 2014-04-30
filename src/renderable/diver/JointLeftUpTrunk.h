#ifndef JOINTLEFTUPTRUNK_H
#define JOINTLEFTUPTRUNK_H

#include "renderable.h"
#include "headers.h"

#include "Joint.h"
#include "LeftForeArm.h"
#include "Dimensions.h"

class JointLeftUpTrunk : public Joint {
    public:
        JointLeftUpTrunk (float radius) : Joint(radius) {
            LeftForeArm *leftForearm = new LeftForeArm(WIDTH_FOREARM, HEIGHT_FOREARM);
            addChild("leftForearm", leftForearm);
            translateChild("leftForearm", 0, 0, -leftForearm->getHeight());
        };

        void drawDownwards (const float * currentTransformationMatrix) {
            glColor3ub(165, 93, 53);
            Joint::drawDownwards(currentTransformationMatrix);
            glColor3ub(255, 255, 255);
        };
};

#endif

