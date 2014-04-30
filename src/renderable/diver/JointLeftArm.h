#ifndef JOINTLEFTUPARM_H
#define JOINTLEFTUPARM_H

#include "renderable.h"
#include "headers.h"

#include "Joint.h"
#include "LeftArm.h"
#include "Dimensions.h"

class JointLeftArm : public Joint {
    public:
        JointLeftArm (float radius) : Joint(radius) {
            LeftArm *leftArm = new LeftArm(WIDTH_ARM, HEIGHT_ARM);
            addChild("leftArm", leftArm);
            translateChild("leftArm", 0, 0, -leftArm->getHeight());
        };

        void drawDownwards (const float * currentTransformationMatrix) {
            glColor3ub(165, 93, 53);
            Joint::drawDownwards(currentTransformationMatrix);
            glColor3ub(255, 255, 255);
        };
};

#endif

