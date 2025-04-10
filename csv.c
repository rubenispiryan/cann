#include <memory.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>
#include <string.h>

#include "csv.h"
// TODO: Currently supports only numeric csv
// TODO: Does not support commas in a cell

#define da_append(vals, val)\
    do {\
        assert(vals);\
        if (vals->count >= vals->capacity) {\
            if (vals->capacity == 0) {\
                vals->capacity = 256;\
            } else {\
                vals->capacity *= 2;\
            }\
            vals->items = realloc(vals->items,\
                                 sizeof(*vals->items) * vals->capacity);\
            assert(vals->items);\
        }\
        vals->items[vals->count] = val;\
        vals->count++;\
    } while (0)\

typedef struct column {
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
        float value = strtof(cell_value, &endptr);
        if (*endptr != '\0') {
            free(buffer);
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

static void csv_print_header(const CSV *csv) {
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

static void csv_print_row(const CSV *csv, int index) {
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