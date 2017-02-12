GCC=gcc
FLAGS = -O3
LIBS = -lfann -lm

TARGET = alphabet_train

all:
	$(GCC) $(FLAGS) $(TARGET).c  -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET) 