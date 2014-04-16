
#include <GL/glew.h>

#include <qapplication.h>
#include "viewer.h"

#include "objLoader/ObjLoader.h"

#include "waves.h"
#include "fog.h"

#include "terrain.h"
#include "shader.h"
#include "SeaDiver.h"
#include "Rectangle.h"
#include <QWidget>

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
        Viewer viewer;
        viewer.setWindowTitle("Sea diver");
        viewer.show();

        //glew initialisation (mandatory)
        log_console.infoStream() << "[Glew Init] " << glewGetErrorString(glewInit());

        Globals::init();
        Globals::print(std::cout);
        Globals::check();
		
		Texture::init();

        log_console.infoStream() << "Running with OpenGL " << Globals::glVersion << " and glsl version " << Globals::glShadingLanguageVersion << " !";

        viewer.setSceneRadius(100.0f);
		
		//EXEMPLE DE COMMENT UTILISER PROGRAM ET TEXTURE DANS TERRAIN.CPP 
		QImage rgb_heightmap = QGLWidget::convertToGLFormat(QImage("img/tamriel3.jpg","jpg"));
        assert(rgb_heightmap.bits());
        unsigned char *black_img = new unsigned char[rgb_heightmap.height()*rgb_heightmap.width()];

        for (int i = 0; i < rgb_heightmap.width(); i++) {
                for (int j = 0; j < rgb_heightmap.height(); j++) {
                        QRgb color = rgb_heightmap.pixel(i,j);
                        black_img[j*rgb_heightmap.width() + i] = (unsigned char) ((qRed(color) + qGreen(color) + qBlue(color))/3);
                }
        }
		
		
		Terrain *terrain = new Terrain(black_img, rgb_heightmap.width(), rgb_heightmap.height(), true);
		terrain->rotate(qglviewer::Quaternion(qglviewer::Vec(1,0,0), 3.14/2));

		Waves *waves = new Waves(0.0,0.0,10.0,10.0,1.0);
		waves->scale(100);

		RenderRoot *root = new RenderRoot();

		unsigned int nParticles = 10000;
		unsigned int nParticleGroups = 1;
		ParticleGroup **pg = new ParticleGroup*[nParticleGroups];
	
		qglviewer::Vec g = 0.01*Vec(0,-9.81,0);
		ParticleGroupKernel *archimede = new ConstantForce(-g);
		ParticleGroupKernel *seaFlow = new SeaFlow(0.2*qglviewer::Vec(1,0,1));
		ParticleGroupKernel *frottement = new FrottementFluideAvance(0.47, 0.47, 0.47, 1000);
		ParticleGroupKernel *attractor = new Attractor(0.2, 100, 0.001);
		ParticleGroupKernel *repulsor = new Attractor(0.1, 0.2, -0.00);
		ParticleGroupKernel *dynamicScheme = new DynamicScheme();
		ParticleGroupKernel *springsSystem = new SpringsSystem(false);

		stringstream name;
	
		for (unsigned int j = 0; j < nParticleGroups; j++) {
			pg[j] = new ParticleGroup(nParticles, 100);
			for (unsigned int i = 0; i < nParticles; i++) {
				qglviewer::Vec pos = Vec(Random::randf(), Random::randf(), Random::randf());
				qglviewer::Vec  vel = Vec(0, 0, 0);
				float r = Random::randf(0.2,0.8);
				float rho = 1.2;//kg/m^3
				float m = rho*4/3.0*3.14*r*r*r;
				pg[j]->addParticle(new Particule(pos, vel, m, r));	
			}
		
			//pg[j]->addKernel(attractor);
			//pg[j]->addKernel(repulsor);
			//pg[j]->addKernel(seaFlow);
			//pg[j]->addKernel(springsSystem);
			pg[j]->addKernel(archimede);
			pg[j]->addKernel(frottement);
			pg[j]->addKernel(dynamicScheme);
			pg[j]->scale(10);
			pg[j]->releaseParticles();
	
			name.clear();
			name << "particules" << j;
			root->addChild(name.str(), pg[j]);
		}


		//root->addChild("terrain", terrain);
		//root->addChild("vagues",[id] waves);

		viewer.addRenderable(root);

        // Run main loop.
        return application.exec();
}

