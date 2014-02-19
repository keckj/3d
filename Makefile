
####################
### LIB EXTERNES ###
####################
OPENCV_LIBPATH = -L/usr/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui

CUDA_INCLUDEPATH = -I/usr/local/cuda-5.5/include
CUDA_LIBPATH = -L/usr/local/cuda-5.5/lib64 
CUDA_LIBS = -lcuda -lcudart

OPENCL_INCLUDEPATH = -I/opt/AMDAPP/include
OPENCL_LIBPATH = -L/opt/AMDAPP/lib/x86
OPENCL_LIBS = -lOpenCL

OPENGL_LIBPATH = -L/usr/X11R6/lib64
OPENGL_INCLUDEPATH = -I/usr/lib64/qt4/mkspecs/linux-g++-64 -I/usr/include/QtCore -I/usr/include/QtGui -I/usr/include/QtOpenGL -I/usr/include/QtXml -I/usr/include -I/usr/X11R6/include 
OPENGL_LIBS = -lglut -lQGLViewer -lGLU -lGL -lQtXml -lQtOpenGL -lQtGui -lQtCore -lpthread 
#OPENGL_LIBS = -lglfw3 -lGL -lGLEW -lGLU -lX11 -lXxf86vm -lXrandr -lpthread -lXi

VIEWER_LIBPATH = -L/usr/X11R6/lib64 -L/usr/lib/x86_64-linux-gnu
VIEWER_INCLUDEPATH = -I/usr/share/mkspecs/linux-g++-64 -I/usr/include/QtCore -I/usr/include/QtGui -I/usr/include/QtOpenGL -I/usr/include/QtXml -I/usr/include -I/usr/include -I/usr/X11R6/include
VIEWER_LIBS = -lQGLViewer -lGLU -lglut -lGL -lQtXml -lQtOpenGL -lQtGui -lQtCore -lpthread 
VIEWER_DEFINES = -D_REENTRANT -DQT_NO_DEBUG -DQT_XML_LIB -DQT_OPENGL_LIB -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED

####################

#Compilateurs
LINK= g++
LINKFLAGS= -W -Wall -Wextra -pedantic -std=c++11
LDFLAGS= $(VIEWER_LIBS)

INCLUDE = -I$(SRCDIR) $(VIEWER_INCLUDEPATH)
LIBS = $(VIEWER_LIBPATH)
DEFINES= $(VIEWER_DEFINES) $(OPT)

CC=gcc
CFLAGS= -W -Wall -Wextra -pedantic -std=c99

CXX=g++
CXXFLAGS= -W -Wall -Wextra -pedantic -std=c++98

NVCC=nvcc
NVCCFLAGS= -arch=sm_20 -Xcompiler -Wall -m64 -O3
CUDADEBUGFLAGS= -Xcompiler -Wall -m64 -G -g -arch=sm_20 

AS = nasm
ASFLAGS= -f elf64

# Autres flags 
DEBUGFLAGS= -g -O0
PROFILINGFLAGS= -pg
RELEASEFLAGS= -O3

# Source et destination des fichiers
TARGET = main

SRCDIR = $(realpath .)/src
OBJDIR = $(realpath .)/obj
EXCL= #excluded dirs in src
EXCLUDED_SUBDIRS = $(foreach DIR, $(EXCL), $(call subdirs, $(SRCDIR)/$(DIR)))
SUBDIRS =  $(filter-out $(EXCLUDED_SUBDIRS), $(call subdirs, $(SRCDIR)))

SRC_EXTENSIONS = c C cc cpp s S asm cu
WEXT = $(addprefix *., $(SRC_EXTENSIONS))
SRC = $(foreach DIR, $(SUBDIRS), $(foreach EXT, $(WEXT), $(wildcard $(DIR)/$(EXT))))
OBJ = $(subst $(SRCDIR), $(OBJDIR), $(addsuffix .o, $(basename $(SRC))))

include rules.mk
