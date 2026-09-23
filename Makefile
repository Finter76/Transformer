CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -Iinclude -D_DEFAULT_SOURCE
LDFLAGS = -lm

BUILD_DIR = build

COMMON_SRC = $(filter-out src/main.c src/train_main.c, $(wildcard src/*.c))
COMMON_OBJ = $(COMMON_SRC:src/%.c=$(BUILD_DIR)/%.o)

RUN_TARGET   = $(BUILD_DIR)/token
TRAIN_TARGET = $(BUILD_DIR)/train

all: $(RUN_TARGET)

$(RUN_TARGET): $(COMMON_OBJ) $(BUILD_DIR)/main.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(TRAIN_TARGET): $(COMMON_OBJ) $(BUILD_DIR)/train_main.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

-include $(wildcard $(BUILD_DIR)/*.d)

clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d $(RUN_TARGET) $(TRAIN_TARGET)

run: $(RUN_TARGET)
	./$(RUN_TARGET)

train: $(TRAIN_TARGET)
	./$(TRAIN_TARGET)

.PHONY: all clean run train
