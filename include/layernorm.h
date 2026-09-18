#ifndef LAYERNORM_H_
#define LAYERNORM_H_

#include "algebra.h"
#include "transformer.h"

Matrix* layer_norm(const Matrix*, const Vector*, const Vector*);

#endif
