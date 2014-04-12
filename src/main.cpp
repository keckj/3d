
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

#include "log.h"
#include "program.h"
#include "globals.h"
#include "cube.h"
#include "cudaUtils.h"
#include "texture.h"
#include "renderRoot.h"


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
		waves->scale(10);

		RenderRoot *root = new RenderRoot();
		root->addChild("terrain", terrain);
		root->addChild("vagues", waves);

		viewer.addRenderable(root);

        // Run main loop.
        return application.exec();
}

