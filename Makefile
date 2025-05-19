CC = clang
CFLAGS = -O3 -std=c11
SDL_CFLAGS = $(shell pkg-config --cflags sdl3)
SDL_LDFLAGS = $(shell pkg-config --libs sdl3)
SOURCES = $(filter-out mnist.c, $(wildcard *.c))
OUTPUT = main

all: $(OUTPUT)

$(OUTPUT): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(OUTPUT)

sdl: $(SOURCES) mnist.c
	$(CC) $(CFLAGS) $(SDL_CFLAGS) $(SOURCES) mnist.c -o $(OUTPUT) $(SDL_LDFLAGS)

clean:
	rm -f $(OUTPUT)