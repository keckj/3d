#include <qapplication.h>
#include "viewer.h"
#include "objLoader/ObjLoader.h"

int main(int argc, char** argv)
{
    // Read command lines arguments.
    QApplication application(argc,argv);

    // Instantiate the viewer.
    Viewer viewer;

    // build your scene here
    viewer.addRenderable(new ObjLoader("obj_files/cube.obj"));

    viewer.setWindowTitle("viewer");
    // Make the viewer window visible on screen.
    viewer.show();

    // Run main loop.
    return application.exec();
}

