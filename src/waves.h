#ifndef _WAVES_H
#define _WAVES_H

#include "renderable.h"
#include "viewer.h"

struct Mobile {
    GLfloat x,y,z; // coordinates
    float vx,vz; // speeds
    GLfloat nx,ny,nz; // normal
    bool isExciter; // is artificially moved
    float frequency; // only in use if isFree == false
    float amplitude; // only in use if isFree == false
    float nextY; // store y coord to be used next time animate() is called
};


class Waves : public Renderable
{
    public:
        ~Waves();
        Waves(float xPos, float zPos, float xWidth, float zWidth, float meanHeight, Viewer *v);
        void draw();
        void animate();
        
    private:
        float xPos, zPos, xWidth, zWidth, meanHeight;

        Viewer *viewer;
        struct Mobile *mobiles;
        unsigned int nMobiles;
        float time;

        GLuint *indices;
        int nIndices;
};
#endif
