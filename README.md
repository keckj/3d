3d
==
Graphique 3D

Compiling:
    WITH CMAKE 3.0+
        mkdir build
        cd build/
        cmake .. 
        make

    WITH THE MAKEFILE (Linux & Mac)
        Edit the top of Makfile 
            L_QGLVIEWER=-lQGLViewer or -lqglviewer to match your QGL_viewer lib
            NARCH=11/20/30 to match your CUDA architecture
        
        make


Executing:
    Execute the generated binary ('main' by default) from the root of the projet.


Minimum CUDA architecture required: 11

Required libraries:  
    OpenAL
    ALUT
    OpenGL
    GLEW
    GLUT
    Qt4 (QtCore QtGui QtXml QtOpenGL)
    QGLViewer
    Log4cpp
    CUDA 5.0+ (6.5 preferred)

 



