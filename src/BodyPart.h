#ifndef BODYPART_H
#define BODYPART_H

#include "renderable.h"

#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

class BodyPart : public Renderable {
    public:
        virtual ~BodyPart();

        void disable ();
        void enable ();
        bool isEnabled () const;

    protected:
        BodyPart (bool enabled = true);
        bool enabled;
};

#endif

