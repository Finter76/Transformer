#ifndef TOKENSEQUENCE_H_
#define TOKENSEQUENCE_H_

typedef struct {
    int *ids;
    int length;
    int capacity;
} TokenSequence;

TokenSequence *token_sequence_create();
int token_sequence_append(TokenSequence*, int);
void token_sequence_free(TokenSequence*);

#endif
