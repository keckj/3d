#include "headers.h"
#include "diver/SeaDiver.h"
#include "diver/Pipe.h"
#include "terrain.h"
#include "SeaDiver.h"
#include "Pipe.h"
#include "Rectangle.h"
#include "log.h"
#include "program.h"
#include "globals.h"
#include "cube.h"
#include "cudaUtils.h"
#include "texture.h"
#include "renderRoot.h"
#include "particleGroup.h"
#include "rand.h"
#include "waves.h"
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
#include "splines/CardinalSpline.h"
#include "skybox.h"

#include <qapplication.h>
#include <QWidget>
#include <vector>
#include <ctime>

#include <ostream>
#include <cassert>
#include <sstream>

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

    // Instantiate the viewer (mandatory)
    Viewer *viewer = new Viewer();
    viewer->setWindowTitle("Sea diver");
    viewer->show();
	Globals::viewer = viewer;

    //glew initialisation (mandatory)
    log_console.infoStream() << "[Glew Init] " << glewGetErrorString(glewInit());
		
	//openal 
	Audible::initOpenALContext();
	alutInit(&argc, argv);
	log_console.infoStream() << "[Alut Init] ";

    Globals::init();
    Globals::print(std::cout);
    Globals::check();

    Texture::init();

    log_console.infoStream() << "Running with OpenGL " << Globals::glVersion << " and glsl version " << Globals::glShadingLanguageVersion << " !";
	//FIN INIT//


	//EXEMPLE DE SON POUR ASSIST
	Audible *test = new Audible("sounds/ambiant/waves_converted.wav", qglviewer::Vec(0,0,0));
	test->setGain(5.0f);
	test->playSource();
	//////////////////////////////////// voir src/utils/openal/audible.h
    
	
	RenderRoot *root = new RenderRoot(); 

	//Terrain
	//Terrain *terrain = new Terrain(black_img, rgb_heightmap.width(), rgb_heightmap.height(), true); 
	//terrain->rotate(qglviewer::Quaternion(qglviewer::Vec(1,0,0), 3.14/2)); 
	//root->addChild("terrain", terrain);


	//Waves
	Waves *waves = new Waves(0.0,0.0,100.0,100.0,10.0);
	waves->scale(10);
	root->addChild("vagues", waves);

    
	// Diver
	SeaDiver *diver = new SeaDiver();
	root->addChild("diver", diver);
	
	//Skybox
    Skybox *skybox = new Skybox();
    viewer->addRenderable(skybox);

    // Pipe
    // TODO : put this in Dimensions
    /* std::vector<Vec> tmp; */
    /* tmp.push_back(Vec(PIPE_FIXED_PART_X, PIPE_FIXED_PART_Y, PIPE_FIXED_PART_Z)); */
    /* tmp.push_back(Vec(0, 2, 4)); */
    /* tmp.push_back(Vec(0, 1, 1)); */
    /* tmp.push_back(Vec(0, 0, 0)); */

    /* Pipe *pipe = new Pipe(tmp); */
    /* tmp.clear(); */
    /* viewer.addRenderable(pipe); */

    

	unsigned int nParticles = 1000;
	unsigned int nLevel = 8;
	ParticleGroup *p = new ParticleGroup(nLevel*nParticles,(nLevel-1)*nParticles);

	qglviewer::Vec g = 0.002*Vec(0,+9.81,0);
	ParticleGroupKernel *archimede = new ConstantForce(g);
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

	p->addKernel(archimede);
	p->addKernel(springsSystem);
	p->addKernel(dynamicScheme);
	p->releaseParticles();
	p->scale(10);
	p->translate(0,10,0);
	root->addChild("particules", p);

	//Configure viwer
	viewer->setSceneRadius(100.0f);
	viewer->addRenderable(root);
	
	//Run main loop.
	application.exec();
	
	alutExit();

	return EXIT_SUCCESS;
}

