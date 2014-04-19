
#include "ressort.h"

Ressort::Ressort(const unsigned int IdP1, unsigned int IdP2,
	   float k, float Lo, float d, float Fmax) :
	IdP1(IdP1), IdP2(IdP2), k(k), Lo(Lo), d(d), Fmax(Fmax)
{
}

