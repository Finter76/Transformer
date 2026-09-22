#include "output.h" 

int choose_token(const Matrix *decoder_output, const Transformer *tr){
    if(!decoder_output || !tr) return -1;

    // Proj
    Matrix *logits = matmul(decoder_output, tr->W_out);
    if(!logits) return -1;    

    int last_row = logits->rows - 1;
    int best_token = 0;
    float best_value = logits->data[last_row * logits->cols];
    for(int i = 1; i < logits->cols; i++){
        float value = logits->data[last_row * logits->cols + i];

        if(value > best_value){
            best_value = value;
            best_token = i;
        }
    }

    mat_free(logits);

    return best_token;
}
