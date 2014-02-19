
# Macros
containing = $(foreach v,$2,$(if $(findstring $1,$v),$v))
not_containing = $(foreach v,$2,$(if $(findstring $1,$v),,$v))
subdirs = $(shell find $1 -type d)

# Règles
all: create_dirs $(TARGET)

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

$(TARGET): $(OBJ)
	@echo
	@echo
	$(LINK) $(LIBS) $^ -o $@ $(LDFLAGS) $(LINKFLAGS) $(DEFINES)
	@echo


$(OBJDIR)%.o : $(SRCDIR)%.c
	@echo
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS) $(DEFINES)

$(OBJDIR)%.o : $(SRCDIR)%.C 
	@echo
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS) $(DEFINES)
$(OBJDIR)%.o : $(SRCDIR)%.cc 
	@echo
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS) $(DEFINES)
$(OBJDIR)%.o : $(SRCDIR)%.cpp 
	@echo
	$(CXX) $(INCLUDE) -o $@ -c $^ $(CXXFLAGS) $(DEFINES)

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
.PHONY: clean cleanall create_dirs

clean:
	-@rm -f $(OBJ)

cleanall:
	-@rm -rf $(TARGET) $(TARGET).out $(OBJDIR)

create_dirs:
	@mkdir -p $(subst $(SRCDIR), $(OBJDIR), $(SUBDIRS))
