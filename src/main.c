#include "config.h"
#include "transformer.h"
#include "tokenizer.h"
#include "embedding.h"
#include "positional.h"
#include "attention.h"
#include "layernorm.h"
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

    /* Multi-Head Self-Attention */
    Matrix *attn_output = attention(X_pe, tr);
    if(!attn_output) return -1;
    print_mat(attn_output); 

    /* Residual Connection */
    Matrix *residual_output = mat_add(X_pe, attn_output);
    if(!residual_output) return -1;
    
    /* Layer Norm */
    Matrix *norm_output = layer_norm(residual_output, tr);
    if(!norm_output) return -1;


    transformer_free(tr);

    return 0;
}
