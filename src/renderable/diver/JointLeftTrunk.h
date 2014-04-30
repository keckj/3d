#ifndef JOINTLEFTTRUNK_H
#define JOINTLEFTTRUNK_H

#include "renderable.h"
#include "headers.h"

#include "Joint.h"
#include "LeftThigh.h"
#include "Dimensions.h"

class JointLeftTrunk : public Joint {
    public:
        JointLeftTrunk (float radius) : Joint(radius) {
            LeftThigh *leftThigh = new LeftThigh(WIDTH_THIGH, HEIGHT_THIGH);
            addChild("leftThigh", leftThigh);
            translateChild("leftThigh", 0, 0, -leftThigh->getHeight());
        };

        void drawDownwards (const float * currentTransformationMatrix) {
            glColor3ub(165, 93, 53);
            Joint::drawDownwards(currentTransformationMatrix);
            glColor3ub(255, 255, 255);
        };
};

#endif

