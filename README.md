
#Graphique 3D

##Compiling:

###Using CMake 3.0 or above

`mkdir build`
`cd build/`
`cmake ..`
`make`

###Using the Makefile (Linux & Mac)
Edit the top of Makfile 
Set `L_QGLVIEWER=-lQGLViewer` or `-lqglviewer` to match your QGLViewer lib
Set `NARCH=11/20/30` to match your CUDA architecture
Compile with `make`


##Executing:
    Execute the generated binary ('main' by default) from the root of the projet.

##Minimum CUDA architecture required: 11

##Required libraries:  
    OpenAL
    ALUT
    OpenGL
    GLEW
    GLUT
    Qt4 (QtCore QtGui QtXml QtOpenGL)
    QGLViewer
    Log4cpp
    CUDA 5.0+ (6.5 preferred)

 



