#include "tokenizer.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

TokenList tokenize(char *p){
    TokenList result = {NULL, 0};
    
    if (!p || p[0] == '\0')
        return result;

    int count = 0;
    char *b = p;
    for(int i = 0; *b != '\0'; i++){
        if(*b == ' ') count++;
        b++;
    }

    count++;
    Token *tokens = malloc(count * sizeof(Token));
    tokens[0].start = p;
    
    int len = 0;
    int word = 0;
    b = p;
    for(int i = 0; *b != '\0'; i++){
        if(*b == ' '){
            tokens[word++].length = len;
            tokens[word].start = b;
            len = 0;
        }
        len++;
        b++;
    } 
    // Last Token
    tokens[word].length = len;

    result.words = count;
    result.tokens = tokens;

    return result;
}

void print_tokens(char *p){
    TokenList phrase = tokenize(p);

    for(int i = 0; i < phrase.words; i++){
        for(int j = 0; j < phrase.tokens[i].length; j++){
            printf("%c", phrase.tokens[i].start[j]);
        }
        printf(" %d", phrase.tokens[i].length);
        printf("\n");
    }
}

void free_tokens(TokenList *t){
    if(!t) return;
    free(t->tokens);
}

int main(){
    char p[] = "Ciao Mi Chiamo Marco";

    TokenList t = tokenize(p);

    print_tokens(p);

    free_tokens(&t);

    return 0;
}
