#ifndef __SKYBOX_H__
#define __SKYBOX_H__

#include "renderTree.h"
#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

class Skybox : public RenderTree {
    public:
        Skybox (float t = 1.0f);
        ~Skybox ();

        bool Initialize ();
        void Render (float camera_yaw, float camera_pitch);
        void Finalize();
		void drawDownwards(const float *currentTransformationMatrix = consts::identity4);

    private:
        void DrawSkyBox (float camera_yaw, float camera_pitch);
        float t;

        GLuint cube_map_texture_ID;
};

#endif

