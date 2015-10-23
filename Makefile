GCC=g++
CFLAGS=-Wall -g -O3
LOADLIBS=-lm

clean:
	rm -rf *.o *.out

test: SRNN.o main.cpp
	$(GCC) $(CFLAGS) -o test SRNN.o main.cpp

SRNN.o: SRNN.hpp SRNN.cpp
	$(GCC) $(CFLAGS) -c SRNN.cpp 
