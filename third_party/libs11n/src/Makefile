#!/usr/bin/make -f
########################################################################
#
# This is a platform-neutral[ish] Makefile for libs11n. It does not
# build the complete shared libraries, but builds the object files for
# use at your leisure.
#
# This makefile does nothing fancy like full depency tracking. It is
# mainly provided to demonstrate what sources go to what parts, to
# assist users who want to get it running under their own build
# environments.
#
# libs11n home page: http://s11n.net
#
########################################################################

default: all

top_srcdir = ..

########################################################################
# You only need to change INCLUDES if you enable some external support,
# like libzfstream (see s11n_config.hpp).
LIBS11N_INCLUDES_DIR = ${top_srcdir}/include
INCLUDES += -I$(LIBS11N_INCLUDES_DIR)

########################################################################
# CXXFLAGS = flags for the C++ compiler.
# Reminder: Debian rules require -fPIC
CXXFLAGS = -Wall -Werror -fPIC -O3

########################################################################
# LDFLAGS_BINS = extra linker args used when building binaries.
# If you've enabled a platform-specific DLL loader, you should add
# its linker arguments here. e.g. if plugin_config.hpp designates
# the use of libdl, then add: -ldl
LDFLAGS_BINS = -rdynamic

########################################################################
# SOURCES_CORE = the code for the core library.
SOURCES_CORE = \
	exception.cpp \
	s11n.cpp \
	s11n_node.cpp

########################################################################
## If you know that s11n has a DLL handler for your platform, set
## the related #defines in the config headers:
##  <s11n.net/s11n/plugin/plugin_config.hpp>
##  <s11n.net/s11n/s11n_config.hpp>
## and link with your appropriate DLL loader library (e.g., libdl or
## libltdl).
##
## By default a no-op handler is used except when WIN32 is defined,
## in which case a LoadModule()-based implementation is used.
## 
## Do not add plugin.PLATFORM.cpp to SOURCES_PLUGIN, as it is included
## directly by plugin.cpp depending on defines from *_config.hpp.
## If you completely disable plugins support (s11n_config.hpp) you can
## unset SOURCES_PLUGIN and build without plugins support.
SOURCES_PLUGIN = \
	path_finder.cpp \
	plugin.cpp


########################################################################
# The SERIALIZER_xxx_SRC defines simplify the inclusion/exclusion
# of individual serializers. For space-limited platforms it can
# be useful to remove all serializers which aren't necessary.
SERIALIZER_compact_SRC = compact.flex.cpp compact_serializer.cpp
SERIALIZER_funtxt_SRC = funtxt.flex.cpp funtxt_serializer.cpp
SERIALIZER_funxml_SRC = funxml.flex.cpp funxml_serializer.cpp
SERIALIZER_simplexml_SRC = simplexml.flex.cpp simplexml_serializer.cpp
SERIALIZER_parens_SRC = parens.flex.cpp parens_serializer.cpp
SERIALIZER_wesnoth_SRC = wesnoth.flex.cpp wesnoth_serializer.cpp

SERIALIZERS_TO_BUILD = \
	$(SERIALIZER_compact_SRC) \
	$(SERIALIZER_funtxt_SRC) \
	$(SERIALIZER_funxml_SRC) \
	$(SERIALIZER_parens_SRC) \
	$(SERIALIZER_simplexml_SRC) \
	$(SERIALIZER_wesnoth_SRC)

ifeq (,$(SERIALIZERS_TO_BUILD))
$(error SERIALIZERS_TO_BUILD must be non-empty)
endif

########################################################################
# SOURCES_IO = the code for the i/o handlers and related code.
SOURCES_IO = \
	data_node_io.cpp \
	strtool.cpp \
	$(SERIALIZERS_TO_BUILD)

foo:
	# SOURCES_IO = $(SOURCES_IO)

########################################################################
# ONLY add expat_serializer.cpp if you have libexpat:
#   SOURCES_IO += expat_serializer.cpp
#   LDFLAGS_BINS += -lexpat
# If enabled, you must link your project against expat and add any
# necessary INCLUDES when compiling THIS code.
########################################################################

########################################################################
# SOURCES_LITE = s11nlite
SOURCES_LITE = \
	s11nlite.cpp


########################################################################
# LIB_SOURCES lists the .cpp files making up the core library, the i/o
# handlers, and s11nlite:
LIB_SOURCES = $(SOURCES_PLUGIN) $(SOURCES_CORE) $(SOURCES_IO) $(SOURCES_LITE)
LIB_OBJECTS = $(patsubst %.cpp,%.o,$(LIB_SOURCES))
LIB_OBJECTS: $(LIB_OBJECTS)
CLEAN_FILES = $(LIB_OBJECTS)

########################################################################
# The main rules...
%.o: %.cpp
	$(CXX) -c $(INCLUDES) $(CPPFLAGS) $(CXXFLAGS) -o $@ $^


########################################################################
# build a statically-linked test/demo app...
#TEST_BIN = test
#TEST_OBJECTS = $(LIB_OBJECTS) test.o
#CLEAN_FILES += test.o
#$(TEST_BIN): $(TEST_OBJECTS)
#	$(CXX) $(LDFLAGS_BINS) -o $@ $(TEST_OBJECTS)

########################################################################
# build a statically-linked s11nconvert...
S11NCONVERT_BIN = s11nconvert
S11NCONVERT_OBJECTS = $(LIB_OBJECTS) main.o argv_parser.o
CLEAN_FILES += main.o argv_parser.o
$(S11NCONVERT_BIN): $(S11NCONVERT_OBJECTS)
	$(CXX) $(LDFLAGS_BINS) -o $@ $(S11NCONVERT_OBJECTS)

########################################################################

########################################################################
# build shared libs11n
LIBS11N_SO = libs11n.so
LIBS11N_SO_VERSION = 1.2
CLEAN_FILES += $(LIBS11N_SO)
libs: $(LIB_OBJECTS)
	$(CXX) $(LDFLAGS) -o $(LIBS11N_SO) -shared --export-dynamic -Wl,-soname=$(LIBS11N_SO).$(LIBS11N_SO_VERSION) $(LIB_OBJECTS)

########################################################################
# Build the binaries:
BINS = $(S11NCONVERT_BIN)
# $(TEST_BIN)
CLEAN_FILES += $(BINS)
bins: LIB_OBJECTS $(BINS)

########################################################################
# Clean up
clean:
	-rm -fr $(CLEAN_FILES)

distclean: clean

########################################################################
# Very basic install rules
prefix ?= /usr/local
DESTDIR ?= 
INSTALL_BINS_DEST = $(DESTDIR)$(prefix)/bin
INSTALL_LIBS_DEST = $(DESTDIR)$(prefix)/lib
INSTALL_HEADERS_DEST = $(DESTDIR)$(prefix)/include
install: bins libs
	@-mkdir -p $(DESTDIR)$(prefix)
	@-mkdir -p $(INSTALL_BINS_DEST) $(INSTALL_LIBS_DEST) $(INSTALL_HEADERS_DEST)
	cp $(S11NCONVERT_BIN) $(INSTALL_BINS_DEST)
	cp $(LIBS11N_SO) $(INSTALL_LIBS_DEST)
	@-rm -fr $(INSTALL_HEADERS_DEST)/s11n.net/s11n
	cp -rp $(LIBS11N_INCLUDES_DIR)/s11n.net $(INSTALL_HEADERS_DEST)

uninstall:
	-rm -f $(INSTALL_BINS_DEST)/$(S11NCONVERT_BIN)
	-rm -f $(INSTALL_LIBS_DEST)/$(LIBS11N_SO)
	-rm -fr $(INSTALL_HEADERS_DEST)/s11n.net/s11n
	-rmdir $(INSTALL_HEADERS_DEST)/s11n.net

########################################################################
# 
all: LIB_OBJECTS
	@echo "Use 'make bins' to attempt to build binaries."; \
	echo "Use 'make libs' to attempt to build shared libraries."

