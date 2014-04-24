#include "skybox.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

#include <QImage>
#include <QGLWidget>

#define GL_TEXTURE_CUBE_MAP_ARB             0x8513
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB  0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB  0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB  0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB  0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB  0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB  0x851A

#include <iostream>

#include "textureCube.h"

Skybox::Skybox (float t) : t(t), textureCube() {
    // Utilisation de la texture CubeMap
    textureCube.bindAndApplyParameters(0);
}

Skybox::~Skybox () {
}

void Skybox::drawDownwards(const float *currentTransformationMatrix) {
    Render(0.0f, 0.0f);
}

void Skybox::Render( float camera_yaw, float camera_pitch ) {
    // Configuration des états OpenGL
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_CUBE_MAP_ARB);
    glDisable(GL_LIGHTING);

    // Désactivation de l'écriture dans le DepthBuffer
    glDepthMask(GL_FALSE);

    // Rendu de la SkyBox
    DrawSkyBox( camera_yaw, camera_pitch );

    // Réactivation de l'écriture dans le DepthBuffer
    glDepthMask(GL_TRUE);

    // Réinitialisation des états OpenGL
    glDisable(GL_TEXTURE_CUBE_MAP_ARB);
    glEnable(GL_LIGHTING);
}

void Skybox::Finalize () {
    // Suppression de la skybox
    glDeleteTextures( 1, &cube_map_texture_ID );
}

void Skybox::DrawSkyBox (float camera_yaw, float camera_pitch) {

    // Rendu de la géométrie
    glBegin(GL_TRIANGLE_STRIP); // X Négatif
    glTexCoord3f(-t,-t,-t); glVertex3f(-t,-t,-t);
    glTexCoord3f(-t,t,-t); glVertex3f(-t,t,-t);
    glTexCoord3f(-t,-t,t); glVertex3f(-t,-t,t);
    glTexCoord3f(-t,t,t); glVertex3f(-t,t,t);
    glEnd();

    glBegin(GL_TRIANGLE_STRIP); // X Positif
    glTexCoord3f(t, -t,-t); glVertex3f(t,-t,-t);
    glTexCoord3f(t,-t,t); glVertex3f(t,-t,t);
    glTexCoord3f(t,t,-t); glVertex3f(t,t,-t);
    glTexCoord3f(t,t,t); glVertex3f(t,t,t);
    glEnd();

    glBegin(GL_TRIANGLE_STRIP); // Y Négatif
    glTexCoord3f(-t,-t,-t); glVertex3f(-t,-t,-t);
    glTexCoord3f(-t,-t,t); glVertex3f(-t,-t,t);
    glTexCoord3f(t, -t,-t); glVertex3f(t,-t,-t);
    glTexCoord3f(t,-t,t); glVertex3f(t,-t,t);
    glEnd();

    glBegin(GL_TRIANGLE_STRIP); // Y Positif
    glTexCoord3f(-t,t,-t); glVertex3f(-t,t,-t);
    glTexCoord3f(t,t,-t); glVertex3f(t,t,-t);
    glTexCoord3f(-t,t,t); glVertex3f(-t,t,t);
    glTexCoord3f(t,t,t); glVertex3f(t,t,t);
    glEnd();

    glBegin(GL_TRIANGLE_STRIP); // Z Négatif
    glTexCoord3f(-t,-t,-t); glVertex3f(-t,-t,-t);
    glTexCoord3f(t, -t,-t); glVertex3f(t,-t,-t);
    glTexCoord3f(-t,t,-t); glVertex3f(-t,t,-t);
    glTexCoord3f(t,t,-t); glVertex3f(t,t,-t);
    glEnd();

    glBegin(GL_TRIANGLE_STRIP); // Z Positif
    glTexCoord3f(-t,-t,t); glVertex3f(-t,-t,t);
    glTexCoord3f(-t,t,t); glVertex3f(-t,t,t);
    glTexCoord3f(t,-t,t); glVertex3f(t,-t,t);
    glTexCoord3f(t,t,t); glVertex3f(t,t,t);
    glEnd();
}

