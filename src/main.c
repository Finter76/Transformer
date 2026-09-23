#include "config.h"
#include "transformer.h"
#include "tokenizer.h"
#include "embedding.h"
#include "positional.h"
#include "encoder.h"
#include "decoder.h"
#include "output.h"
#include "vocab.h"
#include "tokensequence.h"
#include "algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
    srand(42); // Only for training
    Transformer *tr = transformer_init();
    if(!tr) return -1;

    /* Load Italian vocabulary */
    HashTable *table_it = malloc(sizeof(HashTable));
    if(!table_it) return -1;

    hash_table_init(table_it);

    if((vocab_load(table_it, "data/vocab_it.txt") != 0)) return -1;

    /* Load English vocabulary */ 
    HashTable *table_en = malloc(sizeof(HashTable)); 
    if(!table_en) return -1; 

    hash_table_init(table_en); 

    if(vocab_load(table_en, "data/vocab_en.txt") != 0) return -1;

    char phrase[INPUT_BUFFER_SIZE];
    if(!fgets(phrase, INPUT_BUFFER_SIZE, stdin)) return -1;

    TokenList *input_tokens = tokenize(table_it, phrase);
    if(!input_tokens) return -1;
    print_tokens(input_tokens);
 
    /* Token IDs → embedding vectors */
    int *input_ids = token_list_get_ids(input_tokens);
    if(!input_ids) return -1;

    Matrix *X = embed(input_ids, input_tokens->words, tr->E_encoder);
    free(input_ids);
    free_tokens(input_tokens);
    if(!X) return -1;
   
     /* X + positional encoding */
    Matrix *X_pe = positional_encoding(X);
    mat_free(X);
    if(!X_pe) return -1;

    /* Encoder output */
    Matrix *encoder_output = encoder_forward(X_pe, tr);
    mat_free(X_pe);
    if(!encoder_output) return -1;
    //print_mat(encoder_output);    

    TokenSequence *decoder_sequence = token_sequence_create();
    if(!decoder_sequence) return -1;

    int start_id = hash_lookup(table_it, "<BOS>");
    int eos_id   = hash_lookup(table_it, "<EOS>");

    if(start_id == -1 || eos_id == -1) return -1;

    if(token_sequence_append(decoder_sequence, start_id) != 0) return -1;
    
    for(int i = 0; i < MAX_LEN; i++){
        Matrix *decoder_matrix = embed(decoder_sequence->ids, decoder_sequence->length, tr->E_decoder);
        if(!decoder_matrix) return -1;
        
        Matrix *decoder_pe = positional_encoding(decoder_matrix);
        mat_free(decoder_matrix);
        if(!decoder_pe) return -1;
        
        Matrix *decoder_output = decoder_forward(decoder_pe, encoder_output, tr);
        mat_free(decoder_pe);
        if(!decoder_output) return -1;
                
        int next_token = choose_token(decoder_output, tr);
        mat_free(decoder_output);
        if(next_token == -1) return -1;

        const char *word = id_lookup(table_en, next_token);
        printf("%s ", word ? word : "<UNK>");

        token_sequence_append(decoder_sequence, next_token);

        if(next_token == eos_id) break;
     
    }

    mat_free(encoder_output);
    token_sequence_free(decoder_sequence);

    for(int i = 0; i < HASH_TABLE_SIZE; i++){
        free(table_it->entries[i].word);
        free(table_en->entries[i].word);
    }

    free(table_it);
    free(table_en);

    transformer_free(tr);

    return 0;
}
