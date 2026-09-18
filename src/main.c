#include "config.h"
#include "transformer.h"
#include "tokenizer.h"
#include "embedding.h"
#include "positional.h"
#include "encoder.h"
#include "vocab.h"
#include "algebra.h"
#include <stdio.h>
#include <stdlib.h>

int main(){
    Transformer *tr = transformer_init();
    if(!tr) return -1;

    /* Load vocabulary */
    HashTable *table = malloc(sizeof(HashTable));
    if(!table) return -1;

    hash_table_init(table);

    if((vocab_load(table, "vocab.txt") != 0)) return -1;

    char phrase[MAX_LEN];
    if(!fgets(phrase, MAX_LEN, stdin)) return -1;

    TokenList *tokens = tokenize(table, phrase);
    if(!tokens) return -1;
    print_tokens(tokens);
 
    /* Token IDs → embedding vectors */
    Matrix *X = embed(tokens, tr);
    if(!X) return -1;
   
     /* X + positional encoding */
    Matrix *X_pe = positional_encoding(X);
    if(!X_pe) return -1;

    /* Encoder output */
    Matrix *encoder_output = encoder_forward(X_pe, tr);
    if(!encoder_output) return -1;

    print_mat(encoder_output);    

    transformer_free(tr);

    return 0;
}
