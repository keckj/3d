#include "SeaDiver.h"

#include "Trunk.h"
#include "Pipe.h"

#include <iostream>

SeaDiver::SeaDiver() : RenderTree(), dt(0.1), pos(0, 0, 0) {
    Trunk *trunk = new Trunk(WIDTH_TRUNK, HEIGHT_TRUNK, DEPTH_TRUNK);
    addChild("trunk", trunk);

    std::vector<Vec> tmp;
    tmp.push_back(Vec(PIPE_FIXED_PART_X, PIPE_FIXED_PART_Y, PIPE_FIXED_PART_Z));
    tmp.push_back(Vec(0, 2, 4));
    tmp.push_back(Vec(0, 1, 1));
    tmp.push_back(Vec(0, -1, 1));

    Pipe *pipe = new Pipe(tmp);
    addChild("tuyau", pipe);

    // Définition de la trajectoire du plongeur
    std::vector<Vec> trajectoire;
    for (float t = 0; t <= 2 * M_PI; t += 0.1) {
        float x = 6 * cos(t);
        float y = 6 * sin(t);
        trajectoire.push_back(Vec(x, y, 0));
    }

    cs = new CardinalSpline(trajectoire);
}

void SeaDiver::drawDownwards(const float *currentTransformationMatrix) {
    glPushMatrix();

    glMultTransposeMatrixf(relativeModelMatrix);
}

void SeaDiver::drawUpwards(const float *currentTransformationMatrix) {
    glPopMatrix();
}

void SeaDiver::animateDownwards() {
    Vec newPos = (*cs)(t);
    Vec offset =  newPos - pos;
    pos = newPos;

    translate(offset);
    t += dt;
}

// Events
void SeaDiver::keyPressEvent(QKeyEvent* e) {
    // TODO
}

void SeaDiver::mouseMoveEvent(QMouseEvent* e) {
    // TODO
}

