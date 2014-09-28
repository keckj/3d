#ifndef _FOG_H__
#define _FOG_H_

#include "renderable.h"
#include "viewer.h"

class Fog : public Renderable
{
	public:
		Fog(float aboveWaterFogIntensity, float belowWaterFogIntensity, float waterHeight);
		virtual void draw();
		virtual void init(Viewer &);

	private:
		float waterHeight;
        float aboveWaterFogIntensity;
        float belowWaterFogIntensity;
        Viewer *viewer;
        
        // some basic colors...
		static GLfloat black[];
		static GLfloat white[];
		static GLfloat red[];
		static GLfloat blue[];
		static GLfloat green[];
		static GLfloat yellow[];
		static GLfloat magenta[];
		static GLfloat cyan[];
		};

#endif

