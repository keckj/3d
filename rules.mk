
# Macros
containing = $(foreach v,$2,$(if $(findstring $1,$v),$v))
not_containing = $(foreach v,$2,$(if $(findstring $1,$v),,$v))
subdirs = $(shell find $1 -type d)

# Règles

ifeq ($(LINK), NVCC)
debug: LINKFLAGS = $(CUDADEBUGFLAGS) 
else
debug: LINKFLAGS = $(DEBUGFLAGS) 
endif
debug: CFLAGS += $(DEBUGFLAGS)
debug: CXXFLAGS += $(DEBUGFLAGS) 
debug : NVCCFLAGS = $(CUDADEBUGFLAGS)
debug: all

profile: LINKFLAGS += $(PROFILINGFLAGS)
profile: CFLAGS += $(PROFILINGFLAGS)
profile: CXXFLAGS += $(PROFILINGFLAGS)
profile: all

ifeq ($(LINK), NVCC)
else
release: LINKFLAGS += $(RELEASEFLAGS)
endif
release: CFLAGS += $(RELEASEFLAGS)
release: CXXFLAGS += $(RELEASEFLAGS)
release: all

all: create_dirs $(TARGET)

$(TARGET): $(MOCOUTPUT) $(OBJ)
	@echo
	@echo
	$(LINK) $(LIBS) $(OBJ) -o $@ $(LDFLAGS) $(LINKFLAGS) $(DEFINES)
	@echo

#QT macro preprocessor
$(SRCDIR)%.moc : $(SRCDIR)%.hpp
	@echo
	$(MOC) $(INCLUDE) $(DEFINES) -o $@ -i $^ $(MOCFLAGS)
################


$(OBJDIR)%.o : $(SRCDIR)%.c
	$(CC) $(INCLUDE) -o $@ -c $^ $(CFLAGS) $(DEFINES)
	@echo
$(OBJDIR)%.o : $(SRCDIR)%.C 
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS) $(DEFINES)
	@echo
$(OBJDIR)%.o : $(SRCDIR)%.cc 
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS) $(DEFINES)
	@echo
$(OBJDIR)%.o : $(SRCDIR)%.cpp 
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS) $(DEFINES)
	@echo

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
	$(NVCC) $(INCLUDE) -o $@ -c $^ $(NVCCFLAGS) $(DEFINES)


# "-" pour enlever les messages d'erreurs
# "@" pour silent
.PHONY: clean cleanall create_dirs all distrib

clean:
	-@rm -f $(OBJ) 

cleanall:
	-@rm -rf $(TARGET) $(TARGET).out $(OBJDIR) $(MOCOUTPUT)

create_dirs:
	@mkdir -p $(subst $(SRCDIR), $(OBJDIR), $(SUBDIRS))

distrib:
	echo $(DISTRIB)
