#include "Trunk.h"

#include "Head.h"
#include "Dimensions.h"
#include "LeftForeArm.h"
#include "RightForeArm.h"
#include "LeftThigh.h"
#include "RightThigh.h"

#include <iostream>
#include <QGLViewer/vec.h>
using namespace qglviewer;

Trunk::Trunk (float width, float height, float depth) : RenderTree(), rect(width, height, depth) {
    Head *head = new Head(RADIUS_HEAD);
    addChild("head", head);
    translateChild("head", 0, 0, HEIGHT_NECK + getHeight() / 2);

    LeftForeArm *leftForearm = new LeftForeArm(WIDTH_FOREARM, HEIGHT_FOREARM);
    addChild("leftForearm", leftForearm);
    translateChild("leftForearm", 0, leftForearm->getWidth() / 2 + getWidth() / 2, -getHeight() + leftForearm->getHeight());

    RightForeArm *rightForearm = new RightForeArm(WIDTH_FOREARM, HEIGHT_FOREARM);
    addChild("rightForearm", rightForearm);
    translateChild("rightForearm", 0, -rightForearm->getWidth() / 2 - getWidth() / 2, -getHeight() + rightForearm->getHeight());

    LeftThigh *leftThigh = new LeftThigh(WIDTH_THIGH, HEIGHT_THIGH);
    addChild("leftThigh", leftThigh);
    translateChild("leftThigh", 0, (getWidth() - leftThigh->getWidth()) / 2, -getHeight() / 2 - leftThigh->getHeight());

    RightThigh *rightThigh = new RightThigh(WIDTH_THIGH, HEIGHT_THIGH);
    addChild("rightThigh", rightThigh);
    translateChild("rightThigh", 0, (-getWidth() + rightThigh->getWidth()) / 2, -getHeight() / 2 - rightThigh->getHeight());
}

void Trunk::drawDownwards(const float *currentTransformationMatrix) {
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
    rect.draw();
}

void Trunk::drawUpwards (const float *currentTransformationMatrix) {
    glPopMatrix();
}

float Trunk::getWidth () const {
    return rect.getWidth();
}

float Trunk::getHeight () const {
    return rect.getHeight();
}

float Trunk::getDepth () const {
    return rect.getDepth();
}

