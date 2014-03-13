#ifndef TRUNK_H
#define TRUNK_H

#include "BodyPart.h"
#include "cylinder.h"

#define WIDTH_TRUNK 1
#define HEIGHT_TRUNK 2

class Trunk : public BodyPart {
    public:
        Trunk ();
        void draw();
        ~Trunk ();

    private:
        Cylinder cyl;
};

#endif

