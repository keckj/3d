#ifndef TRUNK_H
#define TRUNK_H

#include "BodyPart.h"
#include "Rectangle.h"

class Trunk : public BodyPart {
    public:
        Trunk (float width, float height, float depth);

        float getWidth () const;
        float getHeight() const;
        float getDepth () const;

        void draw();

    private:
        Rectangle rect;
};

#endif

