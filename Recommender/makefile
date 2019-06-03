CC=g++
CFLAGS=-L/usr/local/cuda-9.0/lib64 -lcuda -lcudart

default: Recommender

Recommender: cudacode.o
	$(CC) -std=c++11 -o Recommender *.cpp cuda_functions.o reduction_kernel.o $(CFLAGS) 
cudacode.o:
	nvcc -c cuda_functions.cu reduction_kernel.cu

clean:
	$(RM) Recommender *.o *~	