#ifndef ARM_H
#define ARM_H

#include "../BodyPart.h"
#include "../Cylinder.h"

class Arm : public BodyPart {
    public:
        Arm (float width, float height);

        float getWidth () const;
        float getHeight () const;

        void draw ();

    private:
        Cylinder cyl;
};

#endif

