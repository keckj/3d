#ifndef JOINTLEFTTHIGH_H
#define JOINTLEFTTHIGH_H

#include "renderable.h"
#include "headers.h"

#include "Joint.h"
#include "LeftLeg.h"
#include "Dimensions.h"

class JointLeftThigh : public Joint {
    public:
        JointLeftThigh (float radius) : Joint(radius) {
            LeftLeg *leftLeg = new LeftLeg(WIDTH_LEG, HEIGHT_LEG);
            addChild("leftLeg", leftLeg);
            translateChild("leftLeg", 0, 0, -leftLeg->getHeight());
        };

        void drawDownwards (const float * currentTransformationMatrix) {
            glColor3ub(165, 93, 53);
            Joint::drawDownwards(currentTransformationMatrix);
            glColor3ub(255, 255, 255);
        };
};

#endif

