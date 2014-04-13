#include "RightThigh.h"
#include "RightLeg.h"
#include "Dimensions.h"

RightThigh::RightThigh (float width, float height) : Leg(width, height) {
    RightLeg *rightLeg = new RightLeg(WIDTH_LEG, HEIGHT_LEG);
    addChild("rightLeg", rightLeg);
    translateChild("rightLeg", 0, (WIDTH_TRUNK - getWidth()) / 2, -1.5 * HEIGHT_TRUNK -getHeight());
}

