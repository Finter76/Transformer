#ifndef ATTENTION_H_
#define ATTENTION_H_

#include "config.h"
#include "algebra.h"
#include "transformer.h"

Matrix* attention(Matrix*, Transformer*);
Matrix* masked_attention(Matrix*, Matrix*, Matrix*);

#endif
