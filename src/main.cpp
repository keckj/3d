#include <qapplication.h>
#include "viewer.h"

#include "objLoader/ObjLoader.h"

#include <ctime>
#include "waves.h"
#include "fog.h"

#include "diver/SeaDiver.h"

int main(int argc, char** argv) {

    srand(time(NULL));

    // Read command lines arguments.
    QApplication application(argc,argv);

    // Instantiate the viewer.
    Viewer viewer;
    viewer.setWindowTitle("Sea diver");

    // build your scene here
    //viewer.addRenderable(new ObjLoader("obj_files/cube.obj"));
    viewer.addRenderable(new Waves(0.0f, 0.0f, 50.0f,50.0f,10.0f, &viewer)); 
    viewer.addRenderable(new Fog(0.01f, 0.05f, 10.0f)); 
    viewer.addRenderable(new SeaDiver());


    // Make the viewer window visible on screen.
    viewer.show();

    // Run main loop.
    return application.exec();
}

