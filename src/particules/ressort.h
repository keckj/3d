
#ifndef RESSORT_H
#define RESSORT_H

class Ressort {
    
	public:
        unsigned int IdP1, IdP2;
        float k;
        float Lo;
        float d;
        float Fmax;

        Ressort(const unsigned int IdP1, unsigned int IdP2,
               float k, float Lo, float d, float Fmax);
};

#endif /* end of include guard: RESSORT_H */
