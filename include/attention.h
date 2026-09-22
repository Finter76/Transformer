#ifndef ATTENTION_H_
#define ATTENTION_H_

#include "config.h"
#include "algebra.h"
#include "transformer.h"

Matrix* attention(const Matrix*, const Matrix*, const Matrix*, Matrix *const[], Matrix *const[], Matrix *const[], const Matrix*, int causal);
#endif
