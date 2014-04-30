#ifndef JOINTRIGHTTHIGH_H
#define JOINTRIGHTTHIGH_H

#include "renderable.h"
#include "headers.h"

#include "Joint.h"
#include "RightLeg.h"
#include "Dimensions.h"

class JointRightThigh : public Joint {
    public:
        JointRightThigh (float radius) : Joint(radius) {
            RightLeg *rightLeg = new RightLeg(WIDTH_LEG, HEIGHT_LEG);
            addChild("rightLeg", rightLeg);
            translateChild("rightLeg", 0, 0, -rightLeg->getHeight());
        };

        void drawDownwards (const float * currentTransformationMatrix) {
            glColor3ub(165, 93, 53);
            Joint::drawDownwards(currentTransformationMatrix);
            glColor3ub(255, 255, 255);
        };
};

#endif

