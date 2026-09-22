#ifndef DECODER_H_
#define DECODER_H_

#include "algebra.h"
#include "transformer.h"

Matrix* decoder_forward(const Matrix *X, const Matrix *encoder_output, const Transformer *tr);

#endif
