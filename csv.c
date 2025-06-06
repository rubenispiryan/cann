#include "csv.h"

#include <assert.h>
#include <errno.h>
#include <memory.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
// TODO: Currently supports only numeric csv
// TODO: Does not support commas in a cell

#define da_append(vals, val)                                           \
  do {                                                                 \
    assert(vals);                                                      \
    if (vals->count >= vals->capacity) {                               \
      if (vals->capacity == 0) {                                       \
        vals->capacity = 256;                                          \
      } else {                                                         \
        vals->capacity *= 2;                                           \
      }                                                                \
      vals->items =                                                    \
          realloc(vals->items, sizeof(*vals->items) * vals->capacity); \
      assert(vals->items);                                             \
    }                                                                  \
    vals->items[vals->count] = val;                                    \
    vals->count++;                                                     \
  } while (0)

typedef struct {
  char *col_name_str;
  float *items;
  int count;
  int capacity;
} ColumnDA;

typedef struct csv {
  char *filename_str;
  ColumnDA **items;
  int count;
  int capacity;
  int n_rows;
} CSV;

static const int DEFAULT_CAP = 256;

static char *make_col_name(const char *prefix, int number) {
    assert(prefix);
    int size = snprintf(NULL, 0, "%s_%d", prefix, number) + 1;
    char *result = malloc(size);
    if (result) {
        snprintf(result, size, "%s_%d", prefix, number);
    }
    return result;
}

static ColumnDA *create_column(const char *name_str, int capacity) {
  assert(name_str);
  assert(capacity >= 0);
  float *data = malloc(sizeof(float) * capacity);
  if (data == NULL) {
    return NULL;
  }
  ColumnDA *col = malloc(sizeof(ColumnDA));
  if (col == NULL) {
    free(data);
    return NULL;
  }
  col->items = data;
  col->capacity = capacity;
  col->col_name_str = strdup(name_str);
  col->count = 0;
  if (col->col_name_str == NULL) {
    free(data);
    free(col->col_name_str);
    free(col);
    return NULL;
  }
  return col;
}

static void destroy_column(ColumnDA *col) {
  assert(col);
  free(col->items);
  free(col->col_name_str);
  free(col);
}

static CSV *create_csv(const char *filename_str, int capacity) {
  assert(filename_str);
  assert(capacity >= 0);
  ColumnDA **columns = malloc(sizeof(ColumnDA) * capacity);
  if (columns == NULL) {
    return NULL;
  }
  CSV *csv = malloc(sizeof(CSV));
  if (csv == NULL) {
    free(columns);
    return NULL;
  }
  csv->items = columns;
  csv->capacity = capacity;
  csv->count = 0;
  csv->n_rows = 0;
  csv->filename_str = strdup(filename_str);
  if (csv->filename_str == NULL) {
    free(columns);
    free(csv->filename_str);
    free(csv);
    return NULL;
  }
  return csv;
}

void destroy_csv(CSV *csv) {
  assert(csv);
  for (int i = 0; i < csv->count; i++) {
    destroy_column(csv->items[i]);
  }
  free(csv->items);
  free(csv->filename_str);
  free(csv);
}

static char *csv_read_line(FILE *csv_file) {
  char *buffer = NULL;
  size_t len = 0;
  int buffer_count = 0;
  if ((buffer_count = getline(&buffer, &len, csv_file)) == -1) {
    return NULL;
  }
  // remove new line
  if (buffer[buffer_count - 1] == '\n') {
    buffer[buffer_count - 1] = '\0';
  }
  return buffer;
}

static int parse_header(CSV *csv, FILE *csv_file) {
  assert(csv_file);
  assert(csv);
  char *buffer = csv_read_line(csv_file);
  if (buffer == NULL) {
    return -1;
  }
  char *col_name = strtok(buffer, ",");
  while (col_name != NULL) {
    char *col_name_copy = strdup(col_name);
    if (col_name_copy == NULL) {
      free(buffer);
      return -1;
    }
    ColumnDA *col = create_column(col_name_copy, DEFAULT_CAP);
    da_append(csv, col);
    col_name = strtok(NULL, ",");
  }
  free(buffer);
  return 0;
}

static char *trim(char *str) {
  assert(str);
  while (isspace((unsigned char) *str)) {
    str++;
  }
  if (*str == '\0') {
    return str;
  }
  char *end = str + strlen(str) - 1;
  while (str < end && isspace((unsigned char) *end)) {
    end--;
  }
  *(end + 1) = '\0';
  return str;
}

static int parse_line(CSV *csv, FILE *csv_file) {
  assert(csv_file);
  assert(csv);
  char *buffer = csv_read_line(csv_file);
  if (buffer == NULL) {
    return -1;
  }
  char *cell_value = strtok(buffer, ",");
  int i = 0;
  while (cell_value != NULL) {
    if (i >= csv->count) {
      break;
    }
    ColumnDA *current_col = csv->items[i];
    char *endptr = NULL;
    cell_value = trim(cell_value);
    float value = strtof(cell_value, &endptr);
    if (*endptr != '\0') {
      free(buffer);
      #ifndef DEBUG
        printf("[DEBUG] Could not convert the cell: '%s'\n", cell_value);
        printf("[DEBUG] (int) *endptr: %d\n", *endptr);
      #endif
      return 1;
    }
    da_append(current_col, value);
    cell_value = strtok(NULL, ",");
    i++;
  }
  free(buffer);
  return 0;
}

CSV *read_csv(const char *filename_str) {
  assert(filename_str);
  FILE *csv_file = fopen(filename_str, "r");
  if (csv_file == NULL) {
    fprintf(stderr, "[ERROR] %s (errno: %d)\n", strerror(errno), errno);
    return NULL;
  }
  CSV *csv = create_csv(filename_str, DEFAULT_CAP);
  if (parse_header(csv, csv_file) == -1) {
    destroy_csv(csv);
    fclose(csv_file);
    fprintf(stderr, "[ERROR] Could not parse header\n");
    return NULL;
  }
  int result = 0;
  while ((result = parse_line(csv, csv_file)) != -1) {
    if (result == 1) {
      destroy_csv(csv);
      fclose(csv_file);
      fprintf(stderr, "[ERROR] Found invalid data\n");
      return NULL;
    }
    csv->n_rows++;
  }
  fclose(csv_file);
  return csv;
}

int csv_get_n_rows(const CSV *csv) {
  assert(csv);
  return csv->n_rows;
}

int csv_get_n_cols(const CSV *csv) {
  assert(csv);
  return csv->count;
}

void csv_as_matrix(Matrix *m, const CSV *csv) {
  assert(m);
  assert(csv);
  int n_cols = csv->count;
  int n_rows = csv->n_rows;
  assert(matrix_get_n_cols(m) == n_cols);
  assert(matrix_get_n_rows(m) == n_rows);
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      matrix_set(m, csv->items[j]->items[i], i, j);
    }
  }
}

void csv_row_as_vec(Vector *row, const CSV *csv, int row_i) {
  assert(row);
  assert(csv);
  int n_cols = csv->count;
  assert(vector_get_n(row) == n_cols);
  assert(csv->n_rows > row_i && row_i >= 0);
  for (int i = 0; i < n_cols; i++) {
    vector_set(row, csv->items[i]->items[row_i], i);
  }
}

void csv_col_as_vec(Vector *v, const char *col_name_str, const CSV *csv) {
  assert(v);
  assert(csv);
  assert(col_name_str);
  assert(vector_get_n(v) == csv->n_rows);
  int n_cols = csv->count;
  for (int i = 0; i < n_cols; i++) {
    if (strcmp(col_name_str, csv->items[i]->col_name_str) == 0) {
      vector_copy_data(v, csv->items[i]->items, csv->n_rows);
      return;
    }
  }
}

void csv_cols_as_mat(Matrix *m, const char *const *col_names_str, int cols_len,
                     const CSV *csv) {
  assert(m);
  assert(csv);
  int n_cols = csv->count;
  int n_rows = csv->n_rows;
  assert(matrix_get_n_cols(m) == cols_len);
  assert(matrix_get_n_rows(m) == n_rows);
  assert(cols_len <= n_cols);
  int col_index = 0;
  while (col_index < cols_len) {
    int pre_index = col_index;
    for (int j = 0; j < n_cols; j++) {
      if (strcmp(col_names_str[col_index], csv->items[j]->col_name_str) == 0) {
        for (int i = 0; i < n_rows; i++) {
          matrix_set(m, csv->items[j]->items[i], i, col_index);
        }
        col_index++;
      }
    }
    assert(pre_index < col_index);
  }
}

// TODO: very inefficient
static int *column_unique(ColumnDA *col, int *n_unique, int *rmin) {
    assert(col);
    assert(col->count > 0);
    int count = 1;
    int max = col->items[0];
    int min = col->items[0];
    for (int i = 0; i < col->count; i++) {
        if (col->items[i] > max) {
            max = col->items[i];
        }
        if (col->items[i] < min) {
            min = col->items[i];
        }
    }
    bool *seen = calloc(max + 1 - min, sizeof(bool));
    int *map = calloc(max + 1 - min, sizeof(int));
    if (seen == NULL || map == NULL) {
        return NULL;
    }
    for (int i = 0; i < col->count; i++) {
        int item = col->items[i] - min;
        if (!seen[item]) {
            seen[item] = true;
            map[item] = count;
            count++;
        }
    }
    *n_unique = count - 1;
    free(seen);
    return map;
}

void csv_remove_col_at(CSV *csv, int index) {
    assert(csv);
    int n_cols = csv->count;
    destroy_column(csv->items[index]);
    index++;
    while (index < n_cols) {
      csv->items[index - 1] = csv->items[index];
      index++;
    }
    csv->items[n_cols - 1] = NULL;
    csv->count--;
}

// TODO: reimplement with hash map
void csv_one_hot(CSV *csv, const char *col_name_str) {
    assert(csv);
    assert(col_name_str);
    int n_cols = csv->count;
    int encode_col_i = 0;
    while (encode_col_i < n_cols && strcmp(col_name_str,
           csv->items[encode_col_i]->col_name_str) != 0) {
        encode_col_i++;
    }
    if (encode_col_i == n_cols) {
      return;
    }
    ColumnDA *col = csv->items[encode_col_i];
    int count = 0;
    int min = 0;
    int *map = column_unique(col, &count, &min);
    if (map == NULL) {
        return;
    }
    int start_index = csv->count;
    for (int i = 0; i < count; i++) {
        char *col_name = make_col_name(col->col_name_str, i);
        ColumnDA *new_col = create_column(col_name, col->capacity);
        da_append(csv, new_col);
        free(col_name);
    }
    for (int i = 0; i < col->count; i++) {
        int col_index = map[(int) col->items[i] - min] - 1;
        for (int j = 0; j < count; j++) {
            ColumnDA *new_col = csv->items[start_index + j];
            if (j == col_index) {
                da_append(new_col, 1);
            } else {
                da_append(new_col, 0);
            }
        }
    }
    free(map);
    csv_remove_col_at(csv, encode_col_i);
}

void csv_remove_col(CSV *csv, const char *col_name_str) {
  assert(csv);
  assert(col_name_str);
  int n_cols = csv->count;
  int i = 0;
  while (i < n_cols && strcmp(col_name_str, csv->items[i]->col_name_str) != 0) {
    i++;
  }
  if (i == n_cols) {
    return;
  }
  csv_remove_col_at(csv, i);
}

void csv_print_header(const CSV *csv) {
  assert(csv);
  int n_cols = csv->count;
  for (int i = 0; i < n_cols; i++) {
    if (i > 0) {
      printf(", ");
    }
    printf("%s", csv->items[i]->col_name_str);
  }
  printf("\n");
}

void csv_print_row(const CSV *csv, int index) {
  assert(csv);
  assert(csv->n_rows > index && index >= 0);
  int n_cols = csv->count;
  for (int i = 0; i < n_cols; i++) {
    if (i > 0) {
      printf(", ");
    }
    printf("%.2f", csv->items[i]->items[index]);
  }
  printf("\n");
}

void csv_print(const CSV *csv) {
  assert(csv);
  int n_rows = csv->n_rows;
  csv_print_header(csv);
  for (int i = 0; i < n_rows; i++) {
    csv_print_row(csv, i);
  }
}