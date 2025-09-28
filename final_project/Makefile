CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors -lm

symnmf: symnmf.o symnmf.h
	$(CC) -o symnmf symnmf.o $(CFLAGS)

symnmf.o: symnmf.c
	$(CC) -c symnmf.c $(CFLAGS)

clean:
	rm -f *.o