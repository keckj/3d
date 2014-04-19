
#include "renderRoot.h"
#include "globals.h"
#include "audible.h"

void RenderRoot::drawDownwards(const float *currentTransformationMatrix) {
		Audible::setListenerPosition(Globals::viewer->camera()->position());
		/*Audible::setListenerVelocity(qglviewer::Vec(0,0,0));*/
		/*Audible::setListenerOrientation(qglviewer::Vec(0,0,1), qglviewer::Vec(0,1,0));*/
}

