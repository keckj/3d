#ifndef _CYLINDER_
#define _CYLINDER_

#include "renderable.h"
#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

class Cylinder : public Renderable {
    public:
        Cylinder (float radius, float height);

        float getRadius() const;
        float getHeight() const;

        void draw ();

        ~Cylinder();

    private:
        GLUquadricObj *cyl;
        GLUquadricObj *d;

        float radius;
        float height;
        static int NB_POINTS;
};

#endif

