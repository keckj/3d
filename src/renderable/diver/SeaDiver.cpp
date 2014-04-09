#include "SeaDiver.h"

#include "Pipe.h"

#include "Arm.h"
#include "Trunk.h"
#include "Thigh.h"
#include "Head.h"

SeaDiver::SeaDiver() : Ragdoll() {
    // TODO : put this in Dimensions
    std::vector<Vec> tmp;
    tmp.push_back(Vec(PIPE_FIXED_PART_X, PIPE_FIXED_PART_Y, PIPE_FIXED_PART_Z));
    tmp.push_back(Vec(0, 2, 4));
    tmp.push_back(Vec(0, 1, 1));
    tmp.push_back(Vec(0, 0, 0));

    pipe = new Pipe(tmp);
    tmp.clear();

    trunk = new Trunk(WIDTH_TRUNK, HEIGHT_TRUNK, DEPTH_TRUNK);
    addPart("trunk", trunk);

    leftThigh = new Thigh(WIDTH_THIGH, HEIGHT_THIGH);
    addPart("leftThigh", leftThigh);

    rightThigh = new Thigh(WIDTH_THIGH, HEIGHT_THIGH);
    addPart("rightThigh", rightThigh);

    leftForearm = new Arm(WIDTH_FOREARM, HEIGHT_FOREARM);
    addPart("leftForearm", leftForearm);

    rightForearm = new Arm(WIDTH_FOREARM, HEIGHT_FOREARM);
    addPart("rightForearm", rightForearm);

    head = new Head(RADIUS_HEAD);
    addPart("head", head);
}

void SeaDiver::init (Viewer &viewer) {
    std::cout << "init seadiver" << std::endl;
    pipe->init(viewer);
}

void SeaDiver::draw () {
    // TODO : do better than that
    glPushMatrix();

    pipe->draw();

    getPart("trunk")->draw();

    glPushMatrix();
    glTranslatef(0, 0, HEIGHT_NECK + trunk->getHeight() / 2);
    getPart("head")->draw();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0, -trunk->getWidth() / 2, 0);
    glRotatef(90, 1, 0, 0);
    getPart("leftForearm")->draw();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0, trunk->getWidth() / 2, 0);
    glRotatef(-90, 1, 0, 0);
    getPart("rightForearm")->draw();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0, (trunk->getWidth() - leftThigh->getWidth()) / 2, -1.5 * trunk->getHeight());
    getPart("leftThigh")->draw();
    glPopMatrix();

    glPushMatrix();
    glTranslatef(0, (-trunk->getWidth() + leftThigh->getWidth()) / 2, -1.5 * trunk->getHeight());
    getPart("rightThigh")->draw();
    glPopMatrix();

    glPopMatrix();
}

void SeaDiver::keyPressEvent(QKeyEvent* e, Viewer& viewer) {
    pipe->keyPressEvent(e, viewer);
}

void SeaDiver::mouseMoveEvent(QMouseEvent* e, Viewer& viewer) {
    pipe->mouseMoveEvent(e, viewer);
}

void SeaDiver::animate  () {
    pipe->animate();
}

SeaDiver::~SeaDiver() {
    delete pipe;

    delete head;

    delete leftForearm;
    delete rightForearm;

    delete trunk;

    delete leftThigh;
    delete rightThigh;
}

