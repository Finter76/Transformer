#include "tokenizer.h"
#include "embedding.h"
#include "vocab.h"
#include "algebra.h"
#include <stdio.h>
#include <stdlib.h>

#define MAX_LEN 100

int main(){
    HashTable *table = malloc(sizeof(HashTable));

    hash_table_init(table);

    vocab_load(table, "vocab.txt");

    char phrase[MAX_LEN];

    if(!fgets(phrase, MAX_LEN, stdin)) return 0;

    TokenList *t = tokenize(table, phrase);

    printf("\n");
    print_tokens(t); 
    printf("\n");

    embed(t);

    return 0;
}
