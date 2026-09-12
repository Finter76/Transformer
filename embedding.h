#ifndef EMBEDDING_H_
#define EMBEDDING_H_

#include "tokenizer.h"

// Embeddings Matrix VOCAB_SIZE x VECTOR_SIZE
#define VOCAB_SIZE 1162
#define VECTOR_SIZE 8

void embed(TokenList *tokens); 
void print_embed();

#endif
