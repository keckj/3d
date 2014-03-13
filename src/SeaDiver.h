#ifndef SEADIVER_H
#define SEADIVER_H

#include "Ragdoll.h"
#include "BodyPart.h"

class SeaDiver : public Ragdoll {
    public:
        SeaDiver ();
        ~SeaDiver();

    private:
        BodyPart *trunk;
};

#endif

