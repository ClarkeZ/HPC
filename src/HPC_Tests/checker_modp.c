#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <err.h>
#include <inttypes.h>
#include <getopt.h>
#include <sys/time.h>

#include "mmio.h"

typedef int64_t  i64;
typedef uint64_t u64;
typedef uint32_t u32;

void usage(char **argv)
{
        printf("%s [OPTION...]\n\n", argv[0]);
        printf("Options:\n");
        printf("--matrix FILENAME           sparse matrix file\n");
        printf("--kernel FILENAME           dense block of kernel vectors\n");
        printf("--prime P                   prime modulus\n");
        printf("--right                     verify right kernel vectors\n");
        printf("--left                      verify left kernel vectors [default]\n");
        exit(0);
}

double wtime()
{
        struct timeval ts;
        gettimeofday(&ts, NULL);
        return (double) ts.tv_sec + ts.tv_usec / 1e6;
}

int main(int argc, char **argv)
{
        char *matrix_filename = NULL;
        char *kernel_filename = NULL;
        int n;
        u64 prime = 0;
        bool right = false;

        /* parse command-line options */
        struct option longopts[6] = {
                {"matrix", required_argument, NULL, 'm'},
                {"kernel", required_argument, NULL, 'k'},
                {"prime", required_argument, NULL, 'p'},
                {"right", no_argument, NULL, 'r'},
                {"left", no_argument, NULL, 'l'},
                {NULL, 0, NULL, 0}
        };
        char ch;
        while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
                switch (ch) {
                case 'm':
                        matrix_filename = optarg;
                        break;
                case 'k':
                        kernel_filename = optarg;
                        break;
                case 'p':
                        prime = atoll(optarg);
                        break;
                case 'r':
                        right = true;
                        break;
                case 'l':
                        right = false;
                        break;
                default:
                        errx(1, "Unknown option\n");
                }
        }

        /* argument validation */
        if (matrix_filename == NULL || kernel_filename == NULL || prime == 0)
                usage(argv);

        /* opening the matrix */
        printf("Reading Matrix header from %s\n", matrix_filename);
        FILE * matrix_file = fopen(matrix_filename, "r");
        if (matrix_file == NULL)
                err(1, "cannot open %s", matrix_filename);

        /* read the header, check format */
        int nrows = 0;
        int ncols = 0;
        i64 nnz = 0;
        MM_typecode matcode;
        if (mm_read_banner(matrix_file, &matcode) != 0)
                errx(1, "Could not process Matrix Market banner.\n");
        if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
                errx(1, "Matrix Market type: [%s] not supported (only sparse matrices are OK)", mm_typecode_to_str(matcode));
        if (!mm_is_general(matcode) || !mm_is_integer(matcode))
                errx(1, "Matrix type [%s] not supported (only integer general are OK)", mm_typecode_to_str(matcode));
        if (mm_read_mtx_crd_size(matrix_file, &nrows, &ncols, &nnz) != 0)
                errx(1, "Cannot read matrix size");
        printf("  - [%s] %d x %d with %ld nz\n", mm_typecode_to_str(matcode), nrows, ncols, nnz);

        if (right) { 
                /* implicit transpose */
                int tmp = nrows;
                nrows = ncols;
                ncols = tmp;
        }

        /* open the vector block */
        printf("Reading kernel header from %s\n", kernel_filename);
        FILE * kernel_file = fopen(kernel_filename, "r");
        if (kernel_file == NULL)
                err(1, "cannot open %s", kernel_filename);

        int nk;
        if (mm_read_banner(kernel_file, &matcode) != 0)
                errx(1, "Could not process Matrix Market banner.\n");
        if (!mm_is_matrix(matcode) || !mm_is_array(matcode))
                errx(1, "Matrix Market type: [%s] not supported (only dense matrices are OK)", mm_typecode_to_str(matcode));
        if (!mm_is_general(matcode) || !mm_is_integer(matcode))
                errx(1, "Matrix type [%s] not supported (only integer general are OK)", mm_typecode_to_str(matcode));
        if (mm_read_mtx_array_size(kernel_file, &nk, &n) != 0)
                errx(1, "Cannot read kernel vector block size size");
        printf("  - [%s] %d x %d\n", mm_typecode_to_str(matcode), nk, n);        
        
        if (nk != nrows)
                errx(1, "dimension mismatch");

        /* allocate x and y */
        i64 rblock_size = nrows * n;
        i64 cblock_size = ncols * n;
        i64 tblock_size = rblock_size + cblock_size;
        u32 *x = malloc(sizeof(*x) * rblock_size);
        u32 *y = malloc(sizeof(*y) * cblock_size);
        if (x == NULL || y == NULL)
                errx(1, "cannot allocate vector blocks");
        if (tblock_size > 268435456)
                printf("Allocating %.1f GB\n", tblock_size / 268435456.0);
        else if (tblock_size > 262144)
                printf("Allocating %.1f MB\n", tblock_size / 262144.0);
        else
                printf("Allocating %.1f KB\n", tblock_size / 256.0);
        
        /* load x */
        printf("  - reading kernel vector blocks\n");
        u32 accu[n];
        for (int k = 0; k < n; k++)
                accu[k] = 0;
        for (int k = 0; k < n; k++)
                for (int i = 0; i < nrows; i++) {
                        if (1 != fscanf(kernel_file, "%d\n", &x[i*n + k]))
                                errx(1, "parse error entry %d, %d\n", i, k);
                        if (x[i*n + k] >= prime)
                                errx(1, "entry %d, %d out of bound\n", i, k);
                        accu[k] |= x[i*n + k];
                }

        /* check x != 0 */
        bool ok = false;
        for (int k = 0; k < n; k++)
                if (accu[k] != 0)
                        ok = true;
        if (!ok)
                errx(1, "KO: kernel vectors are all zero");
       
        /* compute y = x*M */
        printf("  - reading matrix + product\n");
        for (int i = 0; i < cblock_size; i++)
                y[i] = 0;
        double start = wtime();
        for (i64 u = 0; u < nnz; u++) {
                int i, j;
                u32 v;
                if (3 != fscanf(matrix_file, "%d %d %d\n", &i, &j, &v))
                        errx(1, "parse error entry %ld\n", u);
                i -= 1;  /* MatrixMarket is 1-based */
                j -= 1;
                u64 vv = v % prime;

                if (right) {  /* on-the-flight transpose */
                        int tmp = i;
                        i = j;
                        j = tmp;
                }

                for (int k = 0; k < n; k++) {
                        u64 xx = x[i*n + k];
                        u64 yy = y[j*n + k];
                        y[j*n + k] = (yy + vv * xx) % prime;
                }

                /* verbosityc*/
                if ((u & 0xfff) == 0xfff) {
                        double elapsed = wtime() - start;
                        double percent = (100. * u) / nnz;
                        double rate = ftell(matrix_file) / 1048576. / elapsed;
                        printf("\r  - Reading %s: %.1f%% (%.1f MB/s)", matrix_filename, percent, rate);
                }
        }
        printf("\n");

        /* check y == 0 */
        for (int i = 0; i < ncols; i++)
                for (int j = 0; j < n; j++)
                        if (y[i * n + j] != 0)
                                errx(1, "KO: y[%d, %d] == %d != 0\n", i, j, y[i * n + j]);
        printf("OK\n");

        exit(EXIT_SUCCESS);
}