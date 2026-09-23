#ifndef CONFIG_H_
#define CONFIG_H_

// Embedding Matrix: VOCAB_SIZE x D_MODEL
#define VOCAB_SIZE_IT 1004
#define VOCAB_SIZE_EN 1003
#define D_MODEL 64 
#define NUM_HEADS 4
#define D_HEAD (D_MODEL/NUM_HEADS)
#define D_FF 256

#define ENCODER_LAYERS 2
#define DECODER_LAYERS 2

#define MAX_LEN 32
#define INPUT_BUFFER_SIZE 256

#define HASH_TABLE_SIZE 2048

#define EPSILON 1e-5

#endif
