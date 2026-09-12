#ifndef TOKENIZER_H_
#define TOKENIZER_H_

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "vocab.h"

typedef struct {
    char *start;
    int length;
    int id;
} Token;

typedef struct {
    Token *tokens;
    int words;
} TokenList;

// Split phrases in words
TokenList* tokenize(HashTable*, char*);

void print_tokens(TokenList*);
void free_tokens(TokenList*);


#endif
