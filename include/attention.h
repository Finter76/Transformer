#ifndef ATTENTION_H_
#define ATTENTION_H_

#include "config.h"
#include "algebra.h"
#include "transformer.h"

Matrix* attention(const Matrix*, const EncoderLayer*);
Matrix* masked_attention(const Matrix*, const DecoderLayer*);

#endif
