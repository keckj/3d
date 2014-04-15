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

bool Skybox::Initialize () {
    // Test de l'extension GL_ARB_texture_cube_map
    char* extensions = (char*) glGetString(GL_EXTENSIONS);

    if(strstr(extensions, "GL_ARB_texture_cube_map") == NULL)
        return false;

    // Liste des faces successives pour la création des textures de CubeMap
    GLenum cube_map_target[6] = {
        GL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB,
        GL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB,
        GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB,
        GL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB
    };

    // Chargement des six textures
    QImage texture_image[6];
    texture_image[0] = QGLWidget::convertToGLFormat(QImage("Skybox/XN.bmp"));
    texture_image[1] = QGLWidget::convertToGLFormat(QImage("Skybox/XP.bmp"));
    texture_image[2] = QGLWidget::convertToGLFormat(QImage("Skybox/YN.bmp"));
    texture_image[3] = QGLWidget::convertToGLFormat(QImage("Skybox/YP.bmp"));
    texture_image[4] = QGLWidget::convertToGLFormat(QImage("Skybox/ZN.bmp"));
    texture_image[5] = QGLWidget::convertToGLFormat(QImage("Skybox/ZP.bmp"));

    // Génération d'une texture CubeMap
    glGenTextures(1, &cube_map_texture_ID);

    // Configuration de la texture
    glBindTexture(GL_TEXTURE_CUBE_MAP_ARB, cube_map_texture_ID);

    for (int i = 0; i < 6; i++) {
        if (texture_image[i].bits()) {
            glTexImage2D(cube_map_target[i], 0, 3, texture_image[i].width(), texture_image[i].height(), 0, GL_RGB, GL_UNSIGNED_BYTE, texture_image[i].bits());
        }
    }

    // Configuration des parametres de la texture
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP );
    glTexParameteri(GL_TEXTURE_CUBE_MAP_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP );

    return true;
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
    // Taille du cube
    float t = 1.0f;

    // Utilisation de la texture CubeMap
    glBindTexture(GL_TEXTURE_CUBE_MAP_ARB, cube_map_texture_ID);

    // Réglage de l'orientation
    glPushMatrix();
    glLoadIdentity();
    glRotatef( camera_pitch, 1.0f, 0.0f, 0.0f );
    glRotatef( camera_yaw, 0.0f, 1.0f, 0.0f );

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

    // Réinitialisation de la matrice ModelView
    glPopMatrix();
}

