#ifndef THIGH_H
#define THIGH_H

#include "BodyPart.h"
#include "../Cylinder.h"

class Thigh : public BodyPart {
    public:
        Thigh (float width, float height);

        float getWidth () const;
        float getHeight () const;

        void draw ();

    private:
        Cylinder cyl;
};

#endif

