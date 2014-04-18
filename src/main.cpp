
#include "headers.h"

#include <qapplication.h>
#include "viewer.h"

#include "objLoader/ObjLoader.h"

#include "waves.h"
#include "fog.h"

#include "terrain.h"
#include "shader.h"
#include "SeaDiver.h"
#include "Rectangle.h"

#include <ostream>
#include <cassert>
#include <sstream>

#include "log.h"
#include "program.h"
#include "globals.h"
#include "cube.h"
#include "cudaUtils.h"
#include "texture.h"
#include "renderRoot.h"
#include "particleGroup.h"
#include "rand.h"
#include "pousseeArchimede.h"
#include "constantForce.h"
#include "constantMassForce.h"
#include "frottementFluide.h"
#include "frottementFluideAvance.h"
#include "dynamicScheme.h"
#include "seaFlow.h"
#include "attractor.h"
#include "springsSystem.h"
#include "audible.h"


using namespace std;
using namespace log4cpp;

int main(int argc, char** argv) {

        srand(time(NULL));

        log4cpp::initLogs();

        CudaUtils::logCudaDevices(log_console);

        log_console.infoStream() << "[Rand Init] ";
        log_console.infoStream() << "[Logs Init] ";

        // glut initialisation (mandatory) 
        glutInit(&argc, argv);
        log_console.infoStream() << "[Glut Init] ";

        // Read command lines arguments.
        QApplication application(argc,argv);
        log_console.infoStream() << "[Qt Init] ";
	

		//openal 
		Audible::initOpenALContext();
		alutInit(&argc, argv);
		log_console.infoStream() << "[Alut Init] ";
		
        // Instantiate the viewer (mandatory)
        Viewer *viewer = new Viewer();
        viewer->setWindowTitle("Sea diver");
        viewer->show();
		Globals::viewer = viewer;

        //glew initialisation (mandatory)
        log_console.infoStream() << "[Glew Init] " << glewGetErrorString(glewInit());

        Globals::init();
        Globals::print(std::cout);
        Globals::check();
		
		Texture::init();

        log_console.infoStream() << "Running with OpenGL " << Globals::glVersion << " and glsl version " << Globals::glShadingLanguageVersion << " !";

	
		//EXEMPLE DE SON POUR ASSIST
		Audible *test = new Audible("sounds/ambiant/waves_converted.wav", qglviewer::Vec(0,0,0));
		test->setGain(5.0f);
		test->playSource();
		//////////////////////////////////// voir src/utils/openal/audible.h
		
		RenderRoot *root = new RenderRoot();

		unsigned int nParticles = 1000;
		unsigned int nLevel = 8;
		ParticleGroup *p = new ParticleGroup(nLevel*nParticles,(nLevel-1)*nParticles);

		qglviewer::Vec g = 0.002*Vec(0,+9.81,0);
		ParticleGroupKernel *archimede = new ConstantForce(g);
		ParticleGroupKernel *seaFlow = new SeaFlow(0.01*qglviewer::Vec(1,0,1));
		ParticleGroupKernel *frottement = new FrottementFluideAvance(0.47, 0.47, 0.47, 1);
		ParticleGroupKernel *attractor = new Attractor(0.2, 100, 0.001);
		ParticleGroupKernel *repulsor = new Attractor(0.1, 1.0, -1.00);
		ParticleGroupKernel *dynamicScheme = new DynamicScheme();
		ParticleGroupKernel *springsSystem = new SpringsSystem(true);

		stringstream name;


		for (unsigned int i = 0; i < nParticles; i++) {
			float r1 = Random::randf(0,5), r2 = Random::randf(0,5);
			for (unsigned int j = 0; j < nLevel; j++) {
				qglviewer::Vec pos = Vec(r1,0.002*j,r2);
				qglviewer::Vec  vel = Vec(Random::randf()*5,0,Random::randf()*5);
				float m = 0.001;
				p->addParticle(new Particule(pos, vel, m, 0.01, j==0));	
			}
		}
		for (unsigned int i = 0; i < nParticles; i++) {
			for (unsigned int j = 0; j < nLevel-1; j++) {
				p->addSpring(nLevel*i+j,nLevel*i+j+1,2,0.1,0.01,0.1);
			}
		}

		//p->addKernel(attractor);
		//p->addKernel(repulsor);
		//p->addKernel(seaFlow);
		p->addKernel(archimede);
		//p->addKernel(frottement);
		p->addKernel(springsSystem);
		p->addKernel(dynamicScheme);
		p->releaseParticles();
		p->scale(10);

		name.clear();
		name << "particules";
		root->addChild(name.str(), p);


        viewer->setSceneRadius(100.0f);
		viewer->addRenderable(root);

		// Run main loop.
		application.exec();

		alutExit();

		return EXIT_SUCCESS;
}

