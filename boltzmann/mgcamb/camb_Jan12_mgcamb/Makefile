
include ${COSMOSIS_SRC_DIR}/config/compilers.mk

USER_FFLAGS=-I.

FISHER=
F90C     = $(FC)
FITSLIB       = cfitsio

ifneq ($(FISHER),)
FFLAGS += -DFISHER
EXTCAMBFILES = Matrix_utils.o
else
EXTCAMBFILES =
endif

include ./Makefile_main
