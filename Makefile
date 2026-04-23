ref/librmrs.so:
	gcc -O2 -shared -fPIC -o ref/librmrs.so \
	    ref/rmrs_wrapper.c \
	    ref/reed_muller.c ref/reed_solomon.c \
	    ref/fft.c ref/gf.c ref/crypto_memset.c \
	    -I ref/

.PHONY: clean
clean:
	rm -f ref/librmrs.so
