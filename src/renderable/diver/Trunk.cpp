#include "Trunk.h"

#include <iostream>

Trunk::Trunk (float width, float height, float depth) : BodyPart(true), rect(width, height, depth) {
}

void Trunk::draw () {
    glPushMatrix();

    rect.draw();

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

