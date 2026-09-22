#include "tokensequence.h"
#include <stddef.h>
#include <stdlib.h>

TokenSequence *token_sequence_create(){
    TokenSequence *s = malloc(sizeof(TokenSequence));
    if(!s) return NULL;

    s->length = 0;
    s->capacity = 16;
    s->ids = malloc(s->capacity * sizeof(int));

    return s;
}

int token_sequence_append(TokenSequence *s, int token_id){
    if(!s) return -1;

    if(s->length >= s->capacity){
        int new_capacity = s->capacity * 2;

        int *new_ids = realloc(s->ids, new_capacity * sizeof(int));
        if(!new_ids)
            return -1;

        s->ids = new_ids;
        s->capacity = new_capacity;
    }

    s->ids[s->length] = token_id;
    s->length++;

    return 0;
}

void token_sequence_free(TokenSequence *s){
    if(!s) return;

    free(s->ids);
    free(s);
}
