#include "config.h"
#include "transformer.h"
#include "tokenizer.h"
#include "vocab.h"
#include "algebra.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
    srand(42); 

    Transformer *tr = transformer_init();
    if(!tr) return -1;

    HashTable *table_it = malloc(sizeof(HashTable));
    if(!table_it) return -1;
    hash_table_init(table_it);
    if(vocab_load(table_it, "data/vocab_it.txt") != 0) return -1;

    HashTable *table_en = malloc(sizeof(HashTable));
    if(!table_en) return -1;
    hash_table_init(table_en);
    if(vocab_load(table_en, "data/vocab_en.txt") != 0) return -1;

    /* Dataset *dataset = dataset_load("data/corpus_it_en.tsv"); */

    /* Training loop */
    /* for(int epoch = 0; epoch < NUM_EPOCHS; epoch++){
     *     for(ogni coppia nel dataset){
     *         forward
     *         loss
     *         backward
     *         optimizer step
     *     }
     *     printf("epoch %d, loss %.4f\n", epoch, avg_loss);
     * }
     */

    /* Salva i pesi allenati (da fare) */
    /* transformer_save(tr, "weights/model.bin"); */

    transformer_free(tr);

    for(int i = 0; i < HASH_TABLE_SIZE; i++){
        free(table_it->entries[i].word);
        free(table_en->entries[i].word);
    }
    free(table_it);
    free(table_en);

    return 0;
}
