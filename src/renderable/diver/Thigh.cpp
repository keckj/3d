#include "Thigh.h"

Thigh::Thigh(float width, float height) : BodyPart(true), cyl(width / 2, height) {
}

float Thigh::getWidth() const {
    return 2 * cyl.getRadius();
}

float Thigh::getHeight() const {
    return cyl.getHeight();
}

void Thigh::draw () {
    glPushMatrix();

    cyl.draw();

    glPopMatrix();
}

