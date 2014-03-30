#include "SeaDiver.h"

#include "Arm.h"
#include "Trunk.h"
#include "Thigh.h"
#include "Head.h"

SeaDiver::SeaDiver() : Ragdoll() {
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

void SeaDiver::draw () {
    // TODO : do better than that
    glPushMatrix();

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

SeaDiver::~SeaDiver() {
    delete head;

    delete leftForearm;
    delete rightForearm;

    delete trunk;

    delete leftThigh;
    delete rightThigh;
}

