#ifndef SEADIVER_H
#define SEADIVER_H

#include "Dimensions.h"

#include "../Ragdoll.h"
#include "../BodyPart.h"

class SeaDiver : public Ragdoll {
    public:
        SeaDiver ();

        void draw ();

        ~SeaDiver();

    private:
        BodyPart *leftForearm, *rightForearm;
        BodyPart *trunk;
        BodyPart *leftThigh, *rightThigh;
};

#endif

