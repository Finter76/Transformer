CC = gcc
CFLAGS = -Wall -Wextra -std=c11
LDFLAGS = -lm

TARGET = build/token

SRC = transformer.c tokenizer.c vocab.c embedding.c algebra.c
OBJ = $(SRC:.c=.o)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

run: $(TARGET)
	./$(TARGET)
