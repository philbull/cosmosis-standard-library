include ${COSMOSIS_SRC_DIR}/config/compilers.mk

PLANCK_DIR=plc-1.0

DEFINES=-D HAS_LAPACK -D LAPACK_CLIK -D NOHEALPIX -D CLIK_LENSING -D 'CLIKSVNVERSION="$(VERSION)"'
USER_CFLAGS=-I$(PLANCK_DIR)/include $(DEFINES)
USER_LDFLAGS=-L$(PLANCK_DIR)/lib -lcosmosis_fortran -lcosmosis

ifeq ($(CFITSIO_INC)$(CFITSIO_LIB),)
$(error To compile the Planck likelihood you must activate the UPS package cfitsio or set the environment variables CFITSIO_INC and CFITSIO_LIB)
endif

ifeq ($(LAPACK_LIB),)
$(error To compile the Planck likelihood you must activate the UPS package lapack or set the environment variable LAPACK_LIB)
endif

all: planck planck_interface.so

planck: $(PLANCK_DIR)
	cd $(PLANCK_DIR) && $(MAKE) install

planck_interface.so: planck planck_interface.c
	$(CC) -shared $(CFLAGS) -lclik -o $@ planck_interface.c $(LDFLAGS) 

clean:
	rm -rf *.o *.so *.dSYM

test:
	@echo Alas, planck has no tests
