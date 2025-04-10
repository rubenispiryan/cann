#ifndef _CSV_HEADER_
#define _CSV_HEADER_

typedef struct csv CSV;

CSV *read_csv(const char *filename_str);

void destroy_csv(CSV *csv);

void csv_print(const CSV *csv);

#endif