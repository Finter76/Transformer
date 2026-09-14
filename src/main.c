#include "config.h"
#include "transformer.h"
#include "tokenizer.h"
#include "embedding.h"
#include "positional.h"
#include "vocab.h"
#include "algebra.h"
#include <stdio.h>
#include <stdlib.h>

int main(){
    Transformer *tr = transformer_init();
    if(!tr) return -1;

    HashTable *table = malloc(sizeof(HashTable));
    if(!table) return -1;
    hash_table_init(table);
    vocab_load(table, "vocab.txt");

    char phrase[MAX_LEN];
    if(!fgets(phrase, MAX_LEN, stdin)) return -1;

    TokenList *t = tokenize(table, phrase);
    print_tokens(t);
 
    Matrix *E = embed(t, tr);
    if(!E) return -1;
   
    Matrix *X = positional_encoding(E);
    if(!X) return -1;
 
    transformer_free(tr);

    return 0;
}
