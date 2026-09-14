#ifndef VOCAB_H_
#define VOCAB_H_

#include "config.h"
#include <stdio.h>
#include <string.h>

typedef struct{
    char *word;
    int id;
} HashNode;

typedef struct{
    HashNode entries[HASH_TABLE_SIZE];
} HashTable;

void hash_table_init(HashTable*);
unsigned long hash(const char*);

void vocab_load(HashTable*, const char*);

void hash_insert(HashTable*, const char*, int);

int hash_lookup(HashTable*, const char*);



#endif
