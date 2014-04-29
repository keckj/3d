#ifndef _WAVES_H
#define _WAVES_H

#include "renderTree.h"
#include "viewer.h"
#include "program.h"
#include "textureCube.h"


struct Mobile {
    GLfloat x,y,z; // coordinates
};

/* ONLY INSTANTIATE ONCE */
class Waves : public RenderTree
{
    public:
        ~Waves();
        Waves(float xPos, float zPos, float xWidth, float zWidth, float meanHeight, TextureCube *cubeMapTexture);
        
    private:
        float xPos, zPos, xWidth, zWidth, meanHeight, deltaX, deltaZ;
		
        Viewer *viewer;
        struct Mobile *mobiles;
        unsigned int nMobiles;
        float time;

        GLuint *indices;
        int nIndices;
        GLuint vertexArray;
        GLuint *vertexBuffers;
        Program program;
		std::map<std::string,int> uniformLocs;

        bool stopAnimating;

        void initializeRelativeModelMatrix();
        
		void drawDownwards(const float *currentTransformationMatrix = consts::identity4);
        void animateDownwards();
        void keyPressEvent(QKeyEvent *e); // Key_P to stop animating waves
};
#endif
