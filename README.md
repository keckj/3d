
#Graphique 3D

Submarine scene demo in OpenGL and CUDA.
Animated bubbles, seeweeds.

Require a CUDA Compute Capability 1.1 and OpenGL 3.3 capable device.

-----

##Compiling:

Everything has been tested with `gcc-4.8` and `gcc-4.9`. 
Other c++0x capable compilers might work as well but have not been tested.

###Using CMake 3.0 or above (preferred method)

```
mkdir build
cd build/
cmake ..
make
```

###Using the Makefile (Linux & Mac)

Edit following variables in `vars.mk` :

- Set `L_QGLVIEWER` to `-lQGLViewer` or `-lqglviewer` to match your QGLViewer library.
- Set `NARCH` to match your device CUDA Compute Capability (minimum 11)
Finally compile with `make`

##Executing:
    Execute the generated binary ('main' by default) from the root of the projet.



##Required libraries:  

Make sure you have all these libraries installed on your computer :

```
    OpenAL
    ALUT
    OpenGL
    GLEW
    GLUT
    Qt4 (QtCore QtGui QtXml QtOpenGL)
    QGLViewer
    Log4cpp
    CUDA 5.0+ (6.5 preferred)
```

Don't forget to add your CUDA library path after a fresh CUDA Toolkit installation.

On linux, simply edit your `~/.bashrc` and add the following lines :
```
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```


