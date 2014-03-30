
OS=$(shell uname -s)

####################
### LIB EXTERNES ###
####################

# Mac
ifeq ($(OS), Darwin)
VIEWER_LIBPATH = -F/usr/local/Cellar/qt/4.8.5/lib -L/usr/local/Cellar/qt/4.8.5/lib  -L/opt/X11/lib -L/usr/local/Cellar/qt/4.8.5/lib -F/usr/local/Cellar/qt/4.8.5/lib
VIEWER_INCLUDEPATH       = -I/usr/local/Cellar/qt/4.8.5/mkspecs/macx-g++ -I. -I/usr/local/Cellar/qt/4.8.5/lib/QtCore.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/lib/QtCore.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/lib/QtGui.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/lib/QtGui.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/lib/QtOpenGL.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/lib/QtOpenGL.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/lib/QtXml.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/lib/QtXml.framework/Versions/4/Headers -I/usr/local/Cellar/qt/4.8.5/include -I/System/Library/Frameworks/OpenGL.framework/Versions/A/Headers -I/System/Library/Frameworks/AGL.framework/Headers -I. -F/usr/local/Cellar/qt/4.8.5/lib
VIEWER_LIBS = -framework Glut -framework QGLViewer -framework OpenGL -framework AGL -framework QtXml -framework QtCore -framework QtOpenGL -framework QtGui 
VIEWER_DEFINES = -D_REENTRANT -DQT_NO_DEBUG -DQT_XML_LIB -DQT_OPENGL_LIB -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED

NARCH=30  #Archi cuda
CUDA_INCLUDEPATH =
CUDA_LIBPATH =
CUDA_LIBS = 
endif

# Linux
ifeq ($(OS), Linux)

VIEWER_LIBPATH = -L/usr/X11R6/lib64 -L/usr/lib/x86_64-linux-gnu
VIEWER_INCLUDEPATH = -I/usr/share/qt4/mkspecs/linux-g++-64 -I/usr/include/QtOpenGL -I/usr/include/QtXml -I/usr/X11R6/include -I/usr/include/qt4/ $(foreach dir, $(shell ls /usr/include/qt4 | xargs), -I/usr/include/qt4/$(dir))
VIEWER_LIBS = -lGLU -lglut -lGL -lQtXml -lQtOpenGL -lQtGui -lQtCore -lpthread
VIEWER_DEFINES = -D_REENTRANT -DQT_NO_DEBUG -DQT_XML_LIB -DQT_OPENGL_LIB -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED

NARCH=30  #Archi cuda
CUDA_INCLUDEPATH = -I/usr/local/cuda-5.5/include
CUDA_LIBPATH = -L/usr/local/cuda-5.5/lib64 
CUDA_LIBS = -lcuda -lcudart

DISTRIB=$(filter-out Distributor ID:, $(shell lsb_release -i))

#Ubuntu
ifeq ($(DISTRIB), Ubuntu)
VIEWER_LIBS+=-lQGLViewer
else #Centos
VIEWER_LIBS+=-lqglviewer
endif

endif

####################

POULPY_LINK= -lGLEW -lopencv_core -lopencv_imgproc -lopencv_highgui

#Compilateurs
LINK= g++
LINKFLAGS= -W -Wall -Wextra -pedantic -std=c++0x
LDFLAGS= $(VIEWER_LIBS) $(POULPY_LINK) #(CUDA_LIBS)
INCLUDE = -I$(SRCDIR) $(foreach dir, $(call subdirs, $(SRCDIR)), -I$(dir)) $(VIEWER_INCLUDEPATH) #$(CUDA_INCLUDEPATH)
LIBS = $(VIEWER_LIBPATH) #$(CUDA_LIBPATH)
DEFINES= $(VIEWER_DEFINES) $(OPT)


CC=gcc
CFLAGS= -W -Wall -Wextra -pedantic -std=c99 -m64

CXX=g++
CXXFLAGS= -W -Wall -Wextra -pedantic -std=c++0x -m64
#-Wshadow -Wstrict-aliasing -Weffc++ -Werror

#preprocesseur QT
MOC=moc 
MOCFLAGS=

NVCC=nvcc
NVCCFLAGS= -Xcompiler -Wall -m64 -gencode arch=compute_$(NARCH),code=sm_$(NARCH) -O3

AS = nasm
ASFLAGS= -f elf64

# Autres flags 
DEBUGFLAGS= -g -O0
CUDADEBUGFLAGS= -Xcompiler -Wall -m64 -G -g -gencode arch=compute_$(NARCH),code=sm_$(NARCH) -Xptxas="-v" 
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

MOCSRC = $(shell grep -rlw $(SRCDIR)/ -e 'Q_OBJECT' --include=*.h | xargs) #need QT preprocessor
MOCOUTPUT = $(addsuffix .moc, $(basename $(MOCSRC)))
SRC = $(foreach DIR, $(SUBDIRS), $(foreach EXT, $(WEXT), $(wildcard $(DIR)/$(EXT))))
OBJ = $(subst $(SRCDIR), $(OBJDIR), $(addsuffix .o, $(basename $(SRC))))

include rules.mk
