#ifndef TOKENIZER_H_
#define TOKENIZER_H_

typedef struct {
    char *start;
    int length;
} Token;

typedef struct {
    Token *tokens;
    int words;
} TokenList;

// Split phrases in words
TokenList tokenize(char*);

void print_tokens(char*);
void free_tokens(TokenList*);


#endif
