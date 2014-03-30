#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

#include <cmath>
#include "fog.h"

// init of colors (static members)
GLfloat Fog::black[]   = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat Fog::white[]   = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat Fog::red[]     = { 1.0f, 0.0f, 0.0f, 1.0f };
GLfloat Fog::green[]   = { 0.0f, 1.0f, 0.0f, 1.0f };
GLfloat Fog::blue[]    = { 0.0f, 0.0f, 1.0f, 1.0f };
GLfloat Fog::yellow[]  = { 1.0f, 1.0f, 0.0f, 1.0f };
GLfloat Fog::magenta[] = { 1.0f, 0.0f, 1.0f, 1.0f };
GLfloat Fog::cyan[]    = { 0.0f, 1.0f, 1.0f, 1.0f };


Fog::Fog(float aboveWaterFogIntensity, float belowWaterFogIntensity, float waterHeight) {
    this->waterHeight = waterHeight;
    this->aboveWaterFogIntensity = aboveWaterFogIntensity;
    this->belowWaterFogIntensity = belowWaterFogIntensity;
}

void Fog::init(Viewer &v) {
    this->viewer = &v;
    
    glDisable(GL_LIGHT0);	
}

void Fog::draw() {
    // Is the camera above or under water?
    if (viewer->camera()->position()[1] < this->waterHeight) {
        glClearColor(0.0f, 0.5f, 0.5f, 1.0f);

        glEnable(GL_FOG);
        GLfloat fogcolor[4] = {0.0,0.5,0.5,1.0};
        glFogi(GL_FOG_MODE, GL_EXP2);
        glFogfv(GL_FOG_COLOR, fogcolor);
        glFogf(GL_FOG_DENSITY, belowWaterFogIntensity);
    } else {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glEnable(GL_FOG);
        GLfloat fogcolor[4] = {1.0,1.0,1.0,1.0};
        glFogi(GL_FOG_MODE, GL_EXP);
        glFogfv(GL_FOG_COLOR, fogcolor);
        glFogf(GL_FOG_DENSITY, aboveWaterFogIntensity);
    }
} 

