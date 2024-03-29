/* Test and timing harness program for developing a dense matrix
   multiplication routine for the CS3014 module */

	#include <stdio.h>
	#include <stdlib.h>
	#include <sys/time.h>
	#include <assert.h>
	#include <pthread.h>
	#include <math.h>
	#include <xmmintrin.h>


/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)

struct complex {
  float real;
  float imag;
};

/* write matrix to stdout */
void write_out(struct complex ** a, int dim1, int dim2)
{
  int i, j;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2 - 1; j++ ) {
      printf("%.3f + %.3fi ", a[i][j].real, a[i][j].imag);
    }
    printf("%.3f + %.3fi\n", a[i][dim2-1].real, a[i][dim2-1].imag);
  }
}


/* create new empty matrix */
struct complex ** new_empty_matrix(int dim1, int dim2)
{
  struct complex ** result = malloc(sizeof(struct complex*) * dim1);
  struct complex * new_matrix = malloc(sizeof(struct complex) * dim1 * dim2);
  int i;

  for ( i = 0; i < dim1; i++ ) {
    result[i] = &(new_matrix[i*dim2]);
  }

  return result;
}

void free_matrix(struct complex ** matrix) {
  free (matrix[0]); /* free the contents */
  free (matrix); /* free the header */
}

/* take a copy of the matrix and return in a newly allocated matrix */
struct complex ** copy_matrix(struct complex ** source_matrix, int dim1, int dim2)
{
  int i, j;
  struct complex ** result = new_empty_matrix(dim1, dim2);

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      result[i][j] = source_matrix[i][j];
    }
  }

  return result;
}

/* create a matrix and fill it with random numbers */
struct complex ** gen_random_matrix(int dim1, int dim2)
{
  const int random_range = 256; // constant power of 2
  struct complex ** result;
  int i, j;
  struct timeval seedtime;
  int seed;

  result = new_empty_matrix(dim1, dim2);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      /* evenly generate values in the range [0, random_range-1)*/
      result[i][j].real = (float)(random() % random_range);
      result[i][j].imag = (float)(random() % random_range);

      /* at no loss of precision, negate the values sometimes */
      /* so the range is now (-(random_range-1), random_range-1)*/
      if (random() & 1) result[i][j].real = -result[i][j].real;
      if (random() & 1) result[i][j].imag = -result[i][j].imag;
    }
  }

  return result;
}

/* check the sum of absolute differences is within reasonable epsilon */
/* returns number of differing values */
void check_result(struct complex ** result, struct complex ** control, int dim1, int dim2)
{
  int i, j;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      double diff;
      diff = abs(control[i][j].real - result[i][j].real);
      sum_abs_diff = sum_abs_diff + diff;

      diff = abs(control[i][j].imag - result[i][j].imag);
      sum_abs_diff = sum_abs_diff + diff;
    }
  }

  if ( sum_abs_diff > EPSILON ) {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
      sum_abs_diff, EPSILON);
  }
}

/* multiply matrix A times matrix B and put result in matrix C */
void matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2)
{
  int i, j, k;

  for ( i = 0; i < a_dim1; i++ ) {
    for( j = 0; j < b_dim2; j++ ) {
      struct complex sum;
      sum.real = 0.0;
      sum.imag = 0.0;
      for ( k = 0; k < a_dim2; k++ ) {
        // the following code does: sum += A[i][k] * B[k][j];
        struct complex product;
        product.real = A[i][k].real * B[k][j].real - A[i][k].imag * B[k][j].imag;
        product.imag = A[i][k].real * B[k][j].imag + A[i][k].imag * B[k][j].real;
        sum.real += product.real;
        sum.imag += product.imag;
      }
      C[i][j] = sum;
    }
  }
}


/*
	This function performs a matrix transposition on A storing in B,
	Starting at position i,j it works on a block of 2 rows x 4 columns
*/
void trans(struct complex** A, struct complex ** B, const int i, const int j, int dim1)
{
	__m128 row1 = _mm_loadu_ps(&A[i][j].real);
	__m128 row2 = _mm_loadu_ps(&A[i][j+2].real);
	__m128 row3 = _mm_loadu_ps(&A[i+1][j].real);
	__m128 row4 = _mm_loadu_ps(&A[i+1][j+2].real);

	__m128 t1 = _mm_shuffle_ps(row1,row2, _MM_SHUFFLE(2,0,2,0));
	__m128 t2 = _mm_shuffle_ps(row1,row2, _MM_SHUFFLE(3,1,3,1));
	__m128 t3 = _mm_shuffle_ps(row3,row4, _MM_SHUFFLE(2,0,2,0));
	__m128 t4 = _mm_shuffle_ps(row3,row4, _MM_SHUFFLE(3,1,3,1));

	_MM_TRANSPOSE4_PS(t1, t2, t3, t4);

	_mm_storeu_ps(&B[j][i].real,t1);
	_mm_storeu_ps(&B[j][i+dim1].real,t2);
	_mm_storeu_ps(&B[j][i+dim1*2].real,t3);
	_mm_storeu_ps(&B[j][i+dim1*3].real,t4);
}


/*	
	This function computes the transpose of A and returns it, 
	it breaks up in blocks of 2 rows by 4 columns and calls the 
	helper function trans above. It handles the edge case by
	redoing the previous elements to accommodate the extra one at the
	end will never be more then 3 columns or 1 row that is redone.
*/
struct complex ** fastTrans(struct complex** A, int dim1, int dim2)
{
	// Minimum 2 rows, 4 columns! i.e 2x4 matrix
	struct complex** res = new_empty_matrix(dim2, dim1);
	int i, j;
	for(i = 0; i<dim1; i+=2)
	{
		for(j = 0; j<dim2; j+=4)
		{
			int t = (j + 4)>dim2? dim2 - 4: j;
			int t2 = (i+2)>dim1? dim1 - 2: i;
			trans(A,res, t2, t, dim1);
		}
	}
	return res;
}


/*
  a struct for the arguments that are passed to each of the threads
*/
struct fmArgs {
	struct complex ** A;
	struct complex ** B;
	struct complex ** C;
	int a_dim1;
	int a_dim2;
	int b_dim2;
	int startA;
	int endA;
};


/*
  Initialise a struct of parameters using the passed parameters.
*/
struct fmArgs * newfmArgs(struct complex ** A, struct complex** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2, int startA, int endA)
{
	struct fmArgs * newA = malloc(sizeof(struct fmArgs));
	newA -> A = A;
	newA -> B = B;
	newA -> C = C;
	newA -> a_dim1 = a_dim1;		//limit for outermost loop
	newA -> a_dim2 = a_dim2;    	//limit for inner loop
	newA -> b_dim2 = b_dim2;    	//limit for middle loop
	newA -> startA = startA;    	//starting value for outermost loop
	newA -> endA = endA;        	//end value for outermost loop

	return newA;
}



/*
	calcElem: this function uses sse vectors in order to multiply a row of A by a column of B. Each thread
	can performs at least one row * column but can complete more depending on the work load.

	The parameters for each thread are stored in a the fmArgs struct.

	The multiplication in this function assumes that B has been transposed. This allows for greater locality.
	So in practice the function is multiplying Arow * Brow rather than the correct Arow * Bcol
	
	It performs two complex number multiplications in a row at at time
*/
void * calcElem(void * a)
{
	struct fmArgs * args = (struct fmArgs *) a;

	__m128 vA, vB, t1, t2, t3, t4, t5, t6, t7, t8, res;
	float * zero __attribute__((aligned(16))) = malloc(sizeof(float)*4);
	float * tres __attribute__((aligned(16))) = malloc(sizeof(float)*4);
	zero[0] = 0.0f;
	zero[1] = 0.0f;
	zero[2] = 0.0f;
	zero[3] = 0.0f;

	//clear the result array before adding the new values to it.
	res = _mm_load_ps(&zero[0]);

	float freal = 0.0f;
	float fimag = 0.0f;

	int i, j, k;
	for(i = args->startA; i<args->endA; i++)
	{
		for(j = 0; j<args->b_dim2; j++)
		{
			//reset result
			res = _mm_load_ps(&zero[0]);
			freal = 0.0f;
			fimag = 0.0f;

			//complex multiplication is done like this (a + bi)(c + di) = (ac-bd) + (ad + bc)i
			for(k=0; k<args->a_dim2; k+= 2)
			{
				//mulitplication is done in groups of 2, so perform this multiplication as long
				//as there is an even number of elements. If there is an odd number, it will be
				//handled in the else statement.
				if(k+2 <= args->a_dim2)
				{
					//load elem1 and elem 2 from A and B
					vA = _mm_loadu_ps(&args->A[i][k].real);
					vB = _mm_loadu_ps(&args->B[j][k].real);

					//swap the real and imaginary value for elem1 and elem2 in B
					t1 = _mm_shuffle_ps(vB, vB,  _MM_SHUFFLE(2,3,0,1));

					//multiply A by B
					t2 = _mm_mul_ps(vA, vB);

					// A x shuffle B (t1)
					t3 = _mm_mul_ps(vA, t1);

					// shuffle t3 and t2 to get ac for elem1 and elem2, followed by ad for elem1 and elem2
					t4 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(2,0,2,0));
					// shuffle t3 and t2 to get bd for elem1 and elem2, followed by bc for elem1 and elem2
					t5 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(3,1,3,1));
					// sub t4 t5 to get ac-bd for elem 1 and elem2, followed by junk
					t6 = _mm_sub_ps(t4, t5);
					// add t4 t5 to get junk, followed by ad+bc for elem1 and elem2
					t7 = _mm_add_ps(t4, t5);
					// get results from t6 and t7
					// t8[0] = ac - bd(1)
					// t8[1] = ac - bd(2)
					// t8[2] = ad + bd(1)
					// t8[3] = ad + bd(2)
					t8 = _mm_shuffle_ps(t6, t7, _MM_SHUFFLE(3,2,1,0));

					//add local result to total
					res = _mm_add_ps(res, t8);
				}
				else
				{
					//this handles the case where the number of elements in the row is odd
					freal = (args->A[i][k].real * args->B[j][k].real)
								- (args->A[i][k].imag * args->B[j][k].imag);
					fimag = (args->A[i][k].real * args->B[j][k].imag)
								+ (args->A[i][k].imag * args->B[j][k].real);
				}
			}

			//tres is used to store res temporarily before they can be added to the non vector total
			_mm_store_ps(tres, res);
			freal += tres[0] + tres[1];
			fimag += tres[2] + tres[3];

			//store results in result array
			args->C[i][j].real = freal;
			args->C[i][j].imag = fimag;
		}
	}
	free(zero);
	free(tres);
	free(args);
	pthread_exit(NULL);
}


#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))

/*
	This performs the same action as the threaded version but performs them on the entire matrix rather
	than a subsection of the matrix. See the other function for a description of how this function
	operates.
*/
void calcElemSerial(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2) {

	__m128 vA, vB, t1, t2, t3, t4, t5, t6, t7, t8, res;
	float * zero __attribute__((aligned(16))) = malloc(sizeof(float)*4);
	float * tres __attribute__((aligned(16))) = malloc(sizeof(float)*4);
	zero[0] = 0.0f;
	zero[1] = 0.0f;
	zero[2] = 0.0f;
	zero[3] = 0.0f;
	res = _mm_load_ps(&zero[0]);

	float freal = 0.0f;
	float fimag = 0.0f;

	struct complex ** transB = fastTrans(B, a_dim2, b_dim2);

	int i, j, k;
	for(i = 0; i<a_dim1; i++)
	{
		for(j = 0; j<b_dim2; j++)
		{
			res = _mm_load_ps(&zero[0]);
			freal = 0.0f;
			fimag = 0.0f;
			for(k=0; k < a_dim2; k+= 2)
			{
				//int t = (k + 2)>args->a_dim2 ? args->a_dim2 - 2: k;

				if(k+2 <= a_dim2)
				{
					vA = _mm_loadu_ps(&A[i][k].real);
					vB = _mm_loadu_ps(&transB[j][k].real);

					//swap real, imag in B
					t1 = _mm_shuffle_ps(vB, vB,  _MM_SHUFFLE(2,3,0,1));
					// A x B
					t2 = _mm_mul_ps(vA, vB);
					// A x shuffle B (t1)
					t3 = _mm_mul_ps(vA, t1);
					// shuffle t3 and t2
					t4 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(2,0,2,0));
					// shuffle t3 and t2
					t5 = _mm_shuffle_ps(t2, t3, _MM_SHUFFLE(3,1,3,1));
					// sub t4 t5
					t6 = _mm_sub_ps(t4, t5);
					// add t4 t5
					t7 = _mm_add_ps(t4, t5);
					// get results from t6 and t7
					// t8[0] = ac - bd(1)
					// t8[1] = ac - bd(2)
					// t8[2] = ad + bd(1)
					// t8[3] = ad + bd(2)
					t8 = _mm_shuffle_ps(t6, t7, _MM_SHUFFLE(3,2,1,0));

					res = _mm_add_ps(res, t8);
				}
				else
				{
					freal = (A[i][k].real * transB[j][k].real)
								- (A[i][k].imag * transB[j][k].imag);
					fimag = (A[i][k].real * transB[j][k].imag)
								+ (A[i][k].imag * transB[j][k].real);
				}
			}
			_mm_store_ps(tres, res);
			freal += tres[0] + tres[1];
			fimag += tres[2] + tres[3];

			C[i][j].real = freal;
			C[i][j].imag = fimag;
		}
	}
	free(zero);
	free(tres);
}

/*
  Breaks the array into sections and passes sections to each thread.
*/
void fasterMul(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2, int numThreads)
{
	struct complex ** transB;
	transB = fastTrans(B, a_dim2, b_dim2); //transpose B to allow for better caching
	pthread_t threads[numThreads];
	struct fmArgs ** args = malloc(sizeof(struct fmArgs)*numThreads);

	/*
	  Divided the work as equally as possible, for algorithm is described in report.
	*/
	int threadIndex = 0;
	int baseInc = a_dim1 / numThreads;
	int suppInc = 1;
	int changeAt = a_dim1 - ((ROUND_UP(a_dim1, numThreads) - a_dim1)*baseInc);
	if(a_dim1 % numThreads == 0)
	{
		suppInc = 0;
	}

	//create and start threads
	int i = 0;
	int count = 0;
	while(i < a_dim1)
	{
		//start thread for i, for baseInc + SuppInc
		args[threadIndex] = newfmArgs(A, transB, C, a_dim1, a_dim2, b_dim2, i, (baseInc + suppInc + i));
		pthread_create(&threads[threadIndex], NULL, calcElem, (void*)args[threadIndex]);
		threadIndex ++;
		i += baseInc + suppInc;

		if(i >= changeAt)
		{
			suppInc = 0;
		}
		count++;
	}

	//wait for threads to finish
	int j;
	for(j=0; j < count; j++)
	{
		pthread_join(threads[j], NULL);
	}
	free(transB[0]);
	free(transB);
	free(args);
}

/*
  This function determines the appropriate function to use based on the sizes of the matrices provided. 
*/
void team_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols) {
	//replace this;
	if((a_rows * a_cols < (16*16)) || (a_rows < 4) || (a_cols < 4)) {
		matmul(A,B,C,a_rows, a_cols, b_cols);
	} else if(a_rows * a_cols < (115 * 115)){
		calcElemSerial(A, B, C, a_rows, a_cols, b_cols);
  } else {
		fasterMul(A, B, C, a_rows, a_cols, b_cols, 64);
  }
}

long long time_diff(struct timeval * start, struct timeval * end) {
  return (end->tv_sec - start->tv_sec) * 1000000L + (end->tv_usec - start->tv_usec);
}

int main(int argc, char ** argv)
{
  struct complex ** A, ** B, ** C;
  struct complex ** control_matrix;
  long long control_time, mul_time;
  double speedup;
  int a_dim1, a_dim2, b_dim1, b_dim2, errs;
  struct timeval pre_time, start_time, stop_time;

  if ( argc != 5 ) {
    fprintf(stderr, "Usage: matmul-harness <A nrows> <A ncols> <B nrows> <B ncols>\n");
    exit(1);
  }
  else {
    a_dim1 = atoi(argv[1]);
    a_dim2 = atoi(argv[2]);
    b_dim1 = atoi(argv[3]);
    b_dim2 = atoi(argv[4]);
  }

  /* check the matrix sizes are compatible */
  if ( a_dim2 != b_dim1 ) {
    fprintf(stderr,
      "FATAL number of columns of A (%d) does not match number of rows of B (%d)\n",
      a_dim2, b_dim1);
    exit(1);
  }

  /* allocate the matrices */
  A = gen_random_matrix(a_dim1, a_dim2);
  B = gen_random_matrix(b_dim1, b_dim2);
  C = new_empty_matrix(a_dim1, b_dim2);
  control_matrix = new_empty_matrix(a_dim1, b_dim2);

  DEBUGGING( {
    printf("matrix A:\n");
    write_out(A, a_dim1, a_dim2);
    printf("\nmatrix B:\n");
    write_out(A, a_dim1, a_dim2);
    printf("\n");
  } )

  /* record control start time */
  gettimeofday(&pre_time, NULL);

  /* use a simple matmul routine to produce control result */
  matmul(A, B, control_matrix, a_dim1, a_dim2, b_dim2);

  /* record starting time */
  gettimeofday(&start_time, NULL);

  /* perform matrix multiplication */
  team_matmul(A, B, C, a_dim1, a_dim2, b_dim2);

  /* record finishing time */
  gettimeofday(&stop_time, NULL);

  /* compute elapsed times and speedup factor */
  control_time = time_diff(&pre_time, &start_time);
  mul_time = time_diff(&start_time, &stop_time);
  speedup = (float) control_time / mul_time;

  printf("Matmul time: %lld microseconds\n", mul_time);
  printf("control time : %lld microseconds\n", control_time);
  if (mul_time > 0 && control_time > 0) {
    printf("speedup: %.2fx\n", speedup);
  }

  /* now check that the team's matmul routine gives the same answer
     as the known working version */
  check_result(C, control_matrix, a_dim1, b_dim2);

  /* free all matrices */
  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
  free_matrix(control_matrix);

  return 0;
}
