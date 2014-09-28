
#Graphique 3D

Submarine scene demo in OpenGL and CUDA.
Animated bubbles, seeweeds.

Require a CUDA Compute Capability 1.1 and OpenGL 3.3 capable device.

##Compiling:

###Using CMake 3.0 or above (preferred method)

```
mkdir build
cd build/
cmake ..
make
```

###Using the Makefile (Linux & Mac)
Edit at the top of Makefile following variables :
    Set `L_QGLVIEWER` to `-lQGLViewer` or `-lqglviewer` to match your QGLViewer lib
    Set `NARCH` to match your CUDA architecture (minimum 11)
Finally compile with `make`

##Executing:
    Execute the generated binary ('main' by default) from the root of the projet.


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


 



