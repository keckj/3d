#ifndef SEADIVER_H
#define SEADIVER_H

#include "Dimensions.h"

#include "Ragdoll.h"
#include "BodyPart.h"
#include "Pipe.h"

class SeaDiver : public Ragdoll {
    public:
        SeaDiver ();

        void init(Viewer &viewer);
        void draw ();
        void animate ();

        // Events
        void keyPressEvent(QKeyEvent* e, Viewer& viewer);
        void mouseMoveEvent(QMouseEvent* e, Viewer& viewer);

        ~SeaDiver();

    private:
        Pipe *pipe;

        BodyPart *head;
        BodyPart *leftForearm, *rightForearm;
        BodyPart *trunk;
        BodyPart *leftThigh, *rightThigh;
};

#endif

