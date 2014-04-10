#ifndef _WAVES_H
#define _WAVES_H

#include "renderable.h"
#include "viewer.h"
#include "program.h"


struct Mobile {
    GLfloat x,y,z; // coordinates
};

/* ONLY INSTANTIATE ONCE */
class Waves : public Renderable
{
    public:
        ~Waves();
        Waves(float xPos, float zPos, float xWidth, float zWidth, float meanHeight);
        void draw();
        void animate();
        void keyPressEvent(QKeyEvent *e, Viewer &v); // Key_P to stop animating waves
        
    private:
        float xPos, zPos, xWidth, zWidth, meanHeight, deltaX, deltaZ;

        Viewer *viewer;
        struct Mobile *mobiles;
        unsigned int nMobiles;
        float time;

        GLuint *indices;
        int nIndices;
        GLuint *vertexBuffers;
        Program program;
        std::vector<int> uniforms_vec;

        bool stopAnimating;

        float *getModelMatrix() const;
};
#endif
