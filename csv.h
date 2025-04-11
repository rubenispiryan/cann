#ifndef _CSV_HEADER_
#define _CSV_HEADER_

#include "matrix.h"
#include "vector.h"

typedef struct csv CSV;

CSV *read_csv(const char *filename_str);

void destroy_csv(CSV *csv);

int csv_get_n_rows(const CSV *csv);

int csv_get_n_cols(const CSV *csv);

void csv_get_row(float *row, int row_size, const CSV *csv, int row_i);

void csv_as_matrix(Matrix *m, const CSV *csv);

void csv_col_as_vec(Vector *v, const char *col_name_str, const CSV *csv);

void csv_remove_col(CSV *csv, const char *col_name_str);

void csv_print_header(const CSV *csv);

void csv_print_row(const CSV *csv, int index);

void csv_print(const CSV *csv);

#endif