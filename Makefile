CC = clang
CFLAGS = -O3 -std=c11
SDL_CFLAGS = $(shell pkg-config --cflags sdl3)
SDL_LDFLAGS = $(shell pkg-config --libs sdl3)
SOURCES = $(filter-out mnist.c, $(wildcard *.c))
OUTPUT = main

all: $(OUTPUT)

$(OUTPUT): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(OUTPUT)

clean:
	rm -f $(OUTPUT)