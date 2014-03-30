#include "Arm.h"

Arm::Arm(float width, float height) : BodyPart(true), cyl(width / 2, height) {
}

float Arm::getWidth() const {
    return 2 * cyl.getRadius();
}

float Arm::getHeight() const {
    return cyl.getHeight();
}

void Arm::draw () {
    glPushMatrix();

    cyl.draw();

    glPopMatrix();
}

