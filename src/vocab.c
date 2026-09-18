#include "vocab.h"
#include <stdlib.h>

void hash_table_init(HashTable *table){
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        table->entries[i].word = NULL;
        table->entries[i].id = -1;
    }
}

unsigned long hash(const char* word){
    unsigned long h = 5381;
    int c;

    while ((c = *word++)) h = ((h << 5) + h) + c;  

    return h;
}

int hash_insert(HashTable *table, const char* word, int id){
    unsigned long index = hash(word) % HASH_TABLE_SIZE;
    unsigned long start = index;

    while (table->entries[index].word != NULL) {
        index = (index + 1) % HASH_TABLE_SIZE;
        if (index == start) return -1;
    }

    table->entries[index].id = id;

    table->entries[index].word = malloc(strlen(word) + 1);
    strcpy(table->entries[index].word, word);

    return 0;
}

int vocab_load(HashTable *table, const char *filename){
    FILE *file = fopen(filename, "r");
    if(!file) return -1;

    char buffer[128];
    int id = 0;

    while (fgets(buffer, sizeof(buffer), file)) {
        buffer[strcspn(buffer, "\n")] = '\0';

        if(hash_insert(table, buffer, id) != 0) return -1;

        id++;
    }

    fclose(file);
    return 0;
}

int hash_lookup(const HashTable *table, const char *word){
    unsigned long index = hash(word) % HASH_TABLE_SIZE;
    unsigned long start = index;

    while (table->entries[index].word != NULL){
        if(strcmp(table->entries[index].word, word) == 0)
            return table->entries[index].id;
        
        index = (index + 1) % HASH_TABLE_SIZE;
    
        if (index == start)
            break;
    }

    return -1;
}
