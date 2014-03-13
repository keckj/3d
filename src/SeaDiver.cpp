#include "SeaDiver.h"

#include "Trunk.h"

SeaDiver::SeaDiver() : Ragdoll() {
    trunk = new Trunk();
    addPart("tronc", trunk);
}

SeaDiver::~SeaDiver() {
    delete trunk;
}

