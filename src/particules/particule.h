#ifndef _PARTICULE_H
#define _PARTICULE_H

#include "headers.h"
#include <iostream>

using namespace qglviewer;  // to use class Vec of the qglviewer lib

class Particule
{
private:
	Vec position;
	Vec velocity;

	double mass; 
	double invMass; // the inverse of the mass is also stored
	double radius;
	
public:
	Particule(Vec pos, Vec vel, double m, double r);
	virtual ~Particule();
	
	const Vec & getPosition() const;
	const Vec & getVelocity() const;

	double getMass() const;
	double getInvMass() const;
	double getRadius() const;

	void setPosition(const Vec &pos);
	void setVelocity(const Vec &vel);
	void incrPosition(const Vec &pos);	// position += pos
	void incrVelocity(const Vec &vel);	// velocity += vel
};

// output stream operator, as non-member
std::ostream& operator<<(std::ostream& os, const Particule& p);

#endif

