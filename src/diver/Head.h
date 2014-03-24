#ifndef HEAD_H
#define HEAD_H

#include "BodyPart.h"

class Head : public BodyPart {
    public:
        Head (float radius);
        float getRadius () const;

        void draw ();

    private:
        float radius;
};

#endif
