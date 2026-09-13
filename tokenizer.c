#include "tokenizer.h"

TokenList* tokenize(HashTable *table, char *p){
    TokenList *result = malloc(sizeof(TokenList));
    
    if (!p || p[0] == '\0')
        return result;
    
    p[strcspn(p, "\n")] = '\0';

    int count = 0;
    char *b = p;

    // Cerca quante parole
    while(*b != '\0'){
        // Fino a che ci sono spazi salta
        while (*b == ' ') b++;

        if (*b == '\0')
           break;

        count++;

        // Fino a quando trovi caratteri sposta il puntatore
        while(*b != ' ' && *b != '\0') b++;
    }

    Token *tokens = malloc(count * sizeof(Token));

    int unk_id = hash_lookup(table, "<UNK>");
    
    int word = 0;
    b = p;
    while(*b != '\0'){
        while (*b == ' ') b++;

        if (*b == '\0')
            break;

        // primo carattere
        tokens[word].start = b;
            
        while (*b != ' ' && *b != '\0')
            b++;
    
        // ultimo carattere
        tokens[word].length = b - tokens[word].start;

        // Token -> Token-ID
        char *buf = malloc(tokens[word].length + 1);
        memcpy(buf, tokens[word].start, tokens[word].length);
        buf[tokens[word].length] = '\0';
            
        int tok_id = hash_lookup(table, buf);
        if(tok_id != -1) tokens[word].id = tok_id;
        else tokens[word].id = unk_id;

        free(buf);
    
        word++;
    }
    result->words = count;
    result->tokens = tokens;

    return result;
}

void print_tokens(TokenList *tl){
    for(int i = 0; i < tl->words; i++){
        for(int j = 0; j < tl->tokens[i].length; j++){
            printf("%c", tl->tokens[i].start[j]);
        }
        printf(" length:%d", tl->tokens[i].length);
        printf(" token id:%d", tl->tokens[i].id);
        printf("\n");
    }
}

void free_tokens(TokenList *t){
    if(!t) return;
    free(t->tokens);
    free(t);
}

