#include "Trunk.h"

#include <iostream>

Trunk::Trunk () : BodyPart(true), cyl(WIDTH_TRUNK, HEIGHT_TRUNK) {
}

void Trunk::draw () {
    glPushMatrix();

    cyl.draw();

    glPopMatrix();
}

Trunk::~Trunk () {
}

