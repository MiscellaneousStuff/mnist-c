#define MAT_AT(m, i, j) ((m).data[(i) * (m).cols + (j)])

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

typedef struct {
    float* data;
    int rows;
    int cols;
} Mat;

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

Mat dot(Mat a, Mat b) {

}

int main(int argc, char *argv[]) {
    srand(42);
    printf("Hello, World!\n");
    Mat a = mat_from((float[]){1, 2, 3,
                                4, 5, 6}, 2, 3);
    mat_print(a);
    return 0;
}