#include "Trunk.h"

#include "Head.h"
#include "Dimensions.h"
#include "JointLeftTrunk.h"
#include "JointRightTrunk.h"
#include "JointLeftUpTrunk.h"
#include "JointRightUpTrunk.h"

#include <iostream>
#include <QGLViewer/vec.h>
using namespace qglviewer;

Trunk::Trunk (float width, float height, float depth) : RenderTree(), rect(width, height, depth) {
    Head *head = new Head(RADIUS_HEAD);
    addChild("head", head);
    translateChild("head", 0, 0, HEIGHT_NECK + getHeight() / 2);
    scaleChild("head", 1, 1, 1.2);

    JointLeftUpTrunk *jointLeftUpTrunk = new JointLeftUpTrunk(RADIUS_JOINT);
    addChild("jointLeftUpTrunk", jointLeftUpTrunk);
    translateChild("jointLeftUpTrunk", 0, getWidth() / 2 + jointLeftUpTrunk->getRadius() / 2, getHeight() / 2 - jointLeftUpTrunk->getRadius());

    JointRightUpTrunk *jointRightUpTrunk = new JointRightUpTrunk(RADIUS_JOINT);
    addChild("jointRightUpTrunk", jointRightUpTrunk);
    translateChild("jointRightUpTrunk", 0, -getWidth() / 2 - jointRightUpTrunk->getRadius() / 2, getHeight() / 2 - jointRightUpTrunk->getRadius());

    JointLeftTrunk *jointLeftTrunk = new JointLeftTrunk(RADIUS_JOINT);
    addChild("jointLeftTrunk", jointLeftTrunk);
    translateChild("jointLeftTrunk", 0, getWidth() / 2 - jointLeftTrunk->getRadius(), -getHeight() / 2);

    JointRightTrunk *jointRightTrunk = new JointRightTrunk(RADIUS_JOINT);
    addChild("jointRightTrunk", jointRightTrunk);
    translateChild("jointRightTrunk", 0, -(getWidth() / 2 - jointRightTrunk->getRadius()), -getHeight() / 2);
}

void Trunk::drawDownwards(const float *currentTransformationMatrix) {
    glPushMatrix();

    glColor3ub(165, 93, 53);
    glMultTransposeMatrixf(relativeModelMatrix);
    rect.draw();
    glColor3ub(255, 255, 255);
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

