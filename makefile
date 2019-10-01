# Makefile

CPPFLAGS = -MMD
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 $(shell sdl2-config --cflags)
LDFLAGS = -export-dynamic $(shell sdl2-config --libs)
LDLIBS = -lm

SRC = main.c function.c

OBJ = ${SRC:.c=.o}

DEP = ${SRC:.c=.d}

all: main

main: ${OBJ}

.PHONY: clean

clean:
	${RM} ${OBJ}
	${RM} ${DEP}
	${RM} main

-include ${DEP}

# END
