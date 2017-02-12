# The makefile requires that the fann library has been built first and is either in the

# GCC=gcc -I ../src/include -L ../src/
GCC=gcc
FLAGS = -O3
LIBS = -lfann -lm

TARGET = large_xor_train

# TARGETS = xor_train xor_test xor_test_fixed simple_train steepness_train simple_test robot mushroom cascade_train scaling_test scaling_train

all:
	$(GCC) $(FLAGS) $(TARGET).c  -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET) 