ref/librmrs.so:
	gcc -O2 -shared -fPIC -o ref/librmrs.so \
	    ref/rmrs_wrapper.c \
	    ref/reed_muller.c ref/reed_solomon.c \
	    ref/fft.c ref/gf.c ref/crypto_memset.c \
	    -I ref/

.PHONY: bench bench-full bench-kem bench-kem-full test clean

bench: ref/librmrs.so
	python3 bench_poly_mul.py

bench-full: ref/librmrs.so
	python3 bench_poly_mul.py --full

bench-kem: ref/librmrs.so
	python3 bench_kem.py

bench-kem-full: ref/librmrs.so
	python3 bench_kem.py --full

test: ref/librmrs.so
	python3 -m pytest tests/ -v

clean:
	rm -f ref/librmrs.so
