#ifndef EMBEDDING_H_
#define EMBEDDING_H_

#include "tokenizer.h"
#include "transformer.h"
#include "algebra.h"

Matrix *embed(const int*, int, const Matrix*);

#endif
