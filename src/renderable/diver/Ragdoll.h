#ifndef RAGDOLL_H
#define RAGDOLL_H

#include "renderable.h"

#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

#include "BodyPart.h"

// Core class of all elements of the sea diver

class Ragdoll : public Renderable {
    public:
        Ragdoll ();

        void addPart (std::string const& name, BodyPart* part);
        BodyPart* getPart (std::string const& name) const;
        void removePart (std::string const& name);

        void disablePart (std::string const& name);
        void enablePart (std::string const& name);

        void draw();
        void animate ();

    protected:
        std::map<std::string, BodyPart*> parts;
};

#endif
