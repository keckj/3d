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
#include "marchingCubes.h"

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
	

	RenderRoot *root = new RenderRoot();
	root->addChild("test", new MarchingCubes());

	//Configure viwer
	viewer->setSceneRadius(100.0f);
	viewer->addRenderable(root);
	
	//Run main loop.
	application.exec();
	
	alutExit();

	return EXIT_SUCCESS;
}

