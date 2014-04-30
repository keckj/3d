#ifndef JOINTRIGHTTRUNK_H
#define JOINTRIGHTTRUNK_H

#include "renderable.h"
#include "headers.h"

#include "Joint.h"
#include "RightThigh.h"
#include "Dimensions.h"

class JointRightTrunk : public Joint {
    public:
        JointRightTrunk (float radius) : Joint(radius) {
            RightThigh *rightThigh = new RightThigh(WIDTH_THIGH, HEIGHT_THIGH);
            addChild("rightThigh", rightThigh);
            translateChild("rightThigh", 0, 0, -rightThigh->getHeight());
        };

        void drawDownwards (const float * currentTransformationMatrix) {
            glColor3ub(165, 93, 53);
            Joint::drawDownwards(currentTransformationMatrix);
            glColor3ub(255, 255, 255);
        };
};

#endif

