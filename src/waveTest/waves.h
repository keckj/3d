#ifndef _WAVES_H
#define _WAVES_H

#include "../renderable.h"
#include "../viewer.h"

struct Mobile {
    GLfloat x,y,z;
    GLfloat nx,ny,nz; // normal
    bool isExciter; // is artificially moved
    float frequency; // only in use if isFree == false
    float amplitude; // only in use if isFree == false
    float speed; // y coord speed
    float nextY; // store y coord to be used next time animate() is called
};


class Waves : public Renderable
{
    public:
        ~Waves();
        Waves(float xWidth, float zWidth, float meanHeight, Viewer *v);
        void draw();
        void animate();
        
    private:
        Viewer *viewer;
        struct Mobile *mobiles;
        unsigned int nMobiles;
        float meanHeight;
        float time;

        GLuint *indices;
        int nIndices;
};
#endif
