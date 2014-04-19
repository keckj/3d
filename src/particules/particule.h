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

	float mass; 
	float invMass; // the inverse of the mass is also stored
	float radius;

	bool fixed;
	
public:
	Particule(Vec pos, Vec vel, float m, float r, bool fixed = false);
	virtual ~Particule();
	
	const Vec & getPosition() const;
	const Vec & getVelocity() const;

	float getMass() const;
	float getInvMass() const;
	float getRadius() const;
	bool isFixed() const;

	void setPosition(const Vec &pos);
	void setVelocity(const Vec &vel);
	void incrPosition(const Vec &pos);	// position += pos
	void incrVelocity(const Vec &vel);	// velocity += vel
};

// output stream operator, as non-member
std::ostream& operator<<(std::ostream& os, const Particule& p);

#endif

