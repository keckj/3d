
#include "renderRoot.h"
#include "globals.h"
#include "audible.h"

void RenderRoot::drawDownwards(const float *currentTransformationMatrix) {
	
		qglviewer::Camera *camera = Globals::viewer->camera();	
		qglviewer::Vec cameraPos = camera->position();
		qglviewer::Vec cameraDir = camera->viewDirection();
		qglviewer::Vec cameraUp = camera->upVector();
		qglviewer::Vec cameraRight = camera->rightVector();

		float proj[16], view[16];
		glGetFloatv(GL_MODELVIEW_MATRIX, view);
		glGetFloatv(GL_PROJECTION_MATRIX, proj);


		GLfloat cameraPos_f[4], cameraDir_f[4], cameraUp_f[4], cameraRight_f[4];
		GLfloat *vectors[] = {cameraPos_f,  cameraDir_f, cameraUp_f, cameraRight_f};
		qglviewer::Vec vec[] = {cameraPos,  cameraDir, cameraUp, cameraRight};

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 3; j++) {
				vectors[i][j] = (GLfloat) vec[i][j];
			}
			vectors[i][3] = 0;
		}

        glBindBuffer(GL_UNIFORM_BUFFER, Globals::projectionViewUniformBlock);
        glBufferSubData(GL_UNIFORM_BUFFER,  0*sizeof(GLfloat), 16*sizeof(GLfloat), proj);
        glBufferSubData(GL_UNIFORM_BUFFER, 16*sizeof(GLfloat), 16*sizeof(GLfloat), view);
        glBufferSubData(GL_UNIFORM_BUFFER, 32*sizeof(GLfloat),  4*sizeof(GLfloat), cameraPos_f);
        glBufferSubData(GL_UNIFORM_BUFFER, 36*sizeof(GLfloat),  4*sizeof(GLfloat), cameraDir_f);
        glBufferSubData(GL_UNIFORM_BUFFER, 40*sizeof(GLfloat),  4*sizeof(GLfloat), cameraUp_f);
        glBufferSubData(GL_UNIFORM_BUFFER, 44*sizeof(GLfloat),  4*sizeof(GLfloat), cameraRight_f);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);

		Audible::setListenerPosition(cameraPos);
		Audible::setListenerVelocity(qglviewer::Vec(0,0,0));
		Audible::setListenerOrientation(cameraDir, cameraUp);
}

