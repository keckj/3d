#ifndef _PARTICLE_
#define _PARTICLE_

#include <QGLViewer/vec.h>
using namespace qglviewer;  // to use class Vec of the qglviewer lib
#include <iostream>

class Particle {
    private:
        Vec position;
        Vec velocity;
        Vec color;
        double mass;
        double invMass; // the inverse of the mass is also stored
        double radius;

    public:
        Particle(Vec pos, Vec vel, double m, double r, Vec color = Vec(1.0, 0.0, 0.0));
        virtual ~Particle();

        const Vec & getPosition() const;
        const Vec & getVelocity() const;
        const Vec & getColor() const;
        double getMass() const;
        double getInvMass() const;
        double getRadius() const;

        void setPosition(const Vec &pos);
        void setVelocity(const Vec &vel);
        void incrPosition(const Vec &pos);  // position += pos
        void incrVelocity(const Vec &vel);  // velocity += vel

        void draw() const;
};

// output stream operator, as non-member
std::ostream& operator<<(std::ostream& os, const Particle& p);

#endif

