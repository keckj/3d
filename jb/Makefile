

### LIBRAIRIES EXTERIEURES ###
OPENCV_INCLUDEPATH = -I/usr/local/include
OPENCV_LIBPATH = -L/usr/lib
OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui

CUDA_INCLUDEPATH = -I/usr/local/cuda-5.5/include
CUDA_LIBPATH = -L/usr/local/cuda-5.5/lib
CUDA_LIBS = -lcuda

OPENCL_INCLUDEPATH = -I/opt/AMDAPP/include
OPENCL_LIBPATH = -L/opt/AMDAPP/lib/x86
OPENCL_LIBS = -lOpenCL

OPENGL_INGLUDEPATH =
OPENGL_LIBPATH =
#OPENGL_LIBS = -lGL -lGLEW -lglfw3 -lX11 -lsfml-graphics -lsfml-window -lsfml-system
OPENGL_LIBS = -lglfw3 -lGL -lGLEW -lGLU -lX11 -lXxf86vm -lXrandr -lpthread -lXi
###############################


#LDFLAGS= -lSDL2main -lSDL2 -lSDL2_ttf -lSDL2_mixer -lGL -lGLU
LDFLAGS= $(OPENGL_LIBS) $(OPENCV_LIBS) -llog4cpp #$(OPENCL_LIBS) $(OPENCV_LIBS) #$(CUDA_LIBS)
INCLUDE = $(OPENGL_INCLUDEPATH)# $(OPENCL_INCLUDEPATH) $(OPENCV_INCLUDEPATH) #$(CUDA_INCLUDEPATH)
LIBS = $(OPENCL_LIBPATH) #$lOpenCL $(OPENGL_LIBPATH) $(OPENCV_LIBPATH) #$(CUDA_LIBPATH)

#Compilateurs
CC=gcc
CFLAGS= -W -Wall -Wextra -pedantic -std=c99

CXX=g++
CXXFLAGS= -W -Wall -Wextra -pedantic -std=c++11

AS = nasm
ASFLAGS= -f elf64

NVCC=nvcc
NVCCFLAGS= -O3 -arch=sm_20 -Xcompiler -Wextra -m64

# Autres règles
DEBUGFLAGS= -g -O0
PROFILINGFLAGS= -pg
RELEASEFLAGS= -O3

# Source et destination des fichiers
SRCDIR = src/
OBJDIR = obj/

TARGET = main

EXT = c C cc cpp s S asm cu
SRC = $(notdir $(wildcard $(addprefix $(SRCDIR)*., $(EXT))))
OBJ = $(addprefix $(OBJDIR), $(addsuffix .o, $(basename $(SRC))))


# Règles
all: $(TARGET)

debug: CFLAGS += $(DEBUGFLAGS)
debug: CXXFLAGS += $(DEBUGFLAGS) 
debug: NVCCFLAGS += $(DEBUGFLAGS) 
debug: $(TARGET)

profile: CFLAGS += $(PROFILINGFLAGS)
profile: CXXFLAGS += $(PROFILINGFLAGS)
profile: NVCCFLAGS += $(PROFILINGFLAGS)
profile: $(TARGET)

release: CFLAGS += $(RELEASEFLAGS)
release: CXXFLAGS += $(RELEASEFLAGS)
release: NVCCFLAGS += $(RELEASEFLAGS)
release: $(TARGET)

$(TARGET): $(OBJ)
	@echo
	@echo
	$(CXX) $(LIBS) $^ -o $@ $(LDFLAGS) $(CXXFLAGS) 
	#$(NVCC) $(LIBS) $(LDFLAGS) $^ -o $@ $(NVCC_FLAGS)
	@echo


$(OBJDIR)%.o : $(SRCDIR)%.c
	@echo
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS)

$(OBJDIR)%.o : $(SRCDIR)%.C 
	@echo
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS)
$(OBJDIR)%.o : $(SRCDIR)%.cc 
	@echo
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS)
$(OBJDIR)%.o : $(SRCDIR)%.cpp 
	@echo
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS)

$(OBJDIR)%.o : $(SRCDIR)%.s
	@echo
	$(AS) $(INCLUDE) -o $@ $^ $(ASFLAGS)
$(OBJDIR)%.o : $(SRCDIR)%.S
	@echo
	$(AS) $(INCLUDE) -o $@ $^ $(ASFLAGS)
$(OBJDIR)%.o : $(SRCDIR)%.asm
	@echo
	$(AS) $(INCLUDE) -o $@ $^ $(ASFLAGS)

$(OBJDIR)%.o: $(SRCDIR)%.cu
	@echo
	$(NVCC) -I $(INCLUDE) -o $@ -c $^ $(NVCCFLAGS)

# "-" pour enlever les messages d'erreurs
# "@" pour silent


.PHONY: clean cleanall

clean:
	-@rm -rf $(OBJDIR)*.o

cleanall: clean
	-@rm -rf $(OBJDIR)

