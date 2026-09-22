#ifndef ENCODER_H_
#define ENCODER_H_

#include "algebra.h"
#include "transformer.h"

Matrix* encoder_forward(const Matrix*, const Transformer *);

#endif
