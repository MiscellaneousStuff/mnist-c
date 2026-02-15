#define MAT_AT(m, i, j) ((m).data[(i) * (m).cols + (j)])

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

typedef struct {
    float* data;
    int rows;
    int cols;
} Mat;

int mat_size(Mat m) {
    return m.rows * m.cols;
}

Mat mat_alloc(int rows, int cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = calloc(rows * cols, sizeof(float));
    return m;
}

Mat mat_from(float* data, int rows, int cols) {
    Mat m = mat_alloc(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        m.data[i] = data[i];
    }
    return m;
}

Mat matmul(Mat m1, Mat m2) {
    // row of A * col of B
    assert(m1.cols == m2.rows);
    int rows = m1.rows;
    int cols = m2.cols;
    Mat m = mat_alloc(rows, cols);
    for (int r=0; r<rows; r++) {
        for (int c=0; c<cols; c++) {
            for (int i=0; i<m1.cols; i++) {
                float a = MAT_AT(m1, r, i);
                float b = MAT_AT(m2, i, c);
                MAT_AT(m, r, c) += a*b;
            }
        };
    }
    return m;
}

Mat mat_clone(Mat m) {
    Mat out = mat_alloc(m.rows, m.cols);
    memcpy(out.data, m.data, m.rows * m.cols * sizeof(float));
    return out;
}

Mat matsub(Mat a, Mat b) {
    assert(a.rows == b.rows);
    assert(a.cols == b.cols);

    Mat m = mat_clone(a);
    for (int i=0; i<mat_size(a); i++) {
        m.data[i] -= b.data[i];
    }
    return m;
}

void mat_print(Mat m) {
    printf("Mat(%d, %d):\n", m.rows, m.cols);
    for (int r = 0; r < m.rows; r++) {
        printf("  [");
        for (int c = 0; c < m.cols; c++) {
            printf("%8.4f", MAT_AT(m, r, c));
            if (c < m.cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

Mat mat_log(Mat x) {
    Mat m = mat_clone(x);
    for (int i=0; i<mat_size(x); i++) {
        m.data[i] = log(x.data[i]);
    }
    return m;
}

Mat mat_exp(Mat x) {
    Mat m = mat_clone(x);
    for (int i=0; i<mat_size(x); i++) {
        m.data[i] = exp(x.data[i]);
    }
    return m;
}

float mat_sum(Mat m) {
    float sum=0;
    for (int i=0; i<mat_size(m); i++)
        sum += m.data[i];
    return sum;
}

Mat sigmoid(Mat x) {
    Mat m = mat_clone(x);
    for (int i=0; i<mat_size(x); i++)
        m.data[i] = 1 / (1 + exp(x.data[i]));
    return m;
}

Mat softmax(Mat x) {
    Mat log_logits = mat_exp(x);
    float s = mat_sum(log_logits);
    for (int i=0; i<mat_size(x); i++) {
        log_logits.data[i] = log_logits.data[i] / s;
    }
    return log_logits;
}

float cross_entropy_loss(Mat pred, Mat target) {
    // x = target * torch.log(pred)
    Mat m = matmul(target, mat_log(pred));

    // x = torch.sum(x) * -1
    return mat_sum(m) * -1;
}

Mat softmax_derivative(Mat pred, Mat target) {
    return matsub(pred, target);
}

typedef struct {
    Mat weights;
    Mat bias;
    int in_dim;
    int out_dim;
} Linear;

Linear init_linear(int in_dim, int out_dim) {
    Linear l;
    float std = sqrt(2.0 / (in_dim + out_dim));
    l.weights = mat_alloc(out_dim, in_dim);
    
    l.bias = mat_alloc(out_dim, 1);
    l.in_dim = in_dim;
    l.out_dim = out_dim;
    return l;
};

Mat linear_forward(Linear l, Mat x) {
    Mat Wx = matmul(l.weights, x);
    Mat Wxb = matadd(Wx, l.bias);
};

void train_xor() {
    Mat x_s = mat_from((float[]){ 0, 0,
                                0, 1,
                                1, 0,
                                1, 1}
                                , 4, 2);
    Mat y_s = mat_from((float[]){ 1, 0,
                                0, 1,
                                0, 1,
                                1, 1}
                                , 4, 2);
    int epochs = 10;
    for (int e=0; e<epochs; e++) {
        Mat x = mat_from(x_s, 1, 2);
        Mat y = mat_from(x_s, 1, 2);
        Mat pred = ?;
        Mat target = y;

        printf("pred:\n");
        mat_print(pred);
        printf("target:\n");
        mat_print(target);

        float loss = cross_entropy_loss(pred, target);
        printf("Epoch := %d, Loss := %0.2f\n", epoch, loss);
    };
};

void misc() {
    Mat a = mat_from((float[]){1, 2, 3,
                                4, 5, 6}, 2, 3);
    Mat b = mat_from((float[]){0.1, 0.2,
                               0.3, 0.4,
                               0.5, 0.6}, 3, 2);
    Mat r = matmul(a, b);
    mat_print(a);
    mat_print(b);
    mat_print(r);
};

int main(int argc, char *argv[]) {
    srand(42);
    printf("Hello, World!\n");
    train_xor();
    return 0;
}