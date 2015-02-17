/* Test and timing harness program for developing a dense matrix
   multiplication routine for the CS3014 module */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <pthread.h>
#include <xmmintrin.h>
#include <math.h>

/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
//#define DEBUGGING(_x) _x
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)

struct complex {
  float real;
  float imag;
};

struct complex ** A;
struct complex ** B;


struct arguments2 {
  struct complex ** C;
  struct complex ** A;
  struct complex ** B;
  int max_k;
  int max_j;
  int start_i;
  int end_i;
};

struct arguments {
  struct complex ** C;
  struct complex ** A;
  struct complex ** B;
  int max_k;
  int max_j;
  int i;
};

struct arguments * new_arguments( struct complex ** A, struct complex ** B, struct complex ** C, int max_k, int max_j, int i) {
  struct arguments * newArgs = malloc(sizeof(struct arguments));
  newArgs->C = C;
  newArgs->A = A;
  newArgs->B = B;
  newArgs->max_k = max_k;
  newArgs->i = i;
  newArgs->max_j = max_j;
  return newArgs;
}

struct arguments2 * new_arguments2( struct complex ** A, struct complex ** B, struct complex ** C, int max_k, int max_j,
                                    int start_i, int end_i) {
  struct arguments2 * newArgs = malloc(sizeof(struct arguments));
  newArgs->C = C;
  newArgs->A = A;
  newArgs->B = B;
  newArgs->max_k = max_k;
  newArgs->start_i = start_i;
  newArgs->end_i = end_i;
  newArgs->max_j = max_j;
  return newArgs;
}

void * calculate_element(void * args) {
  struct arguments * arguments = (struct arguments *) args;
  struct complex sum;
  sum.real = 0.0;
  sum.imag = 0.0;
  float *dest = malloc(sizeof(float)*4);
  int k,j;
  __m128 aPart;
  __m128 bPart;
  __m128 res;
  struct complex product;
  product.real = 0.0;
  product.imag = 0.0;
  for(j = 0; j < arguments->max_j; j++) {
    struct complex product;
    sum.real = 0.0;
    sum.imag = 0.0;
    for ( k = 0; k < arguments->max_k; k++ ) {
      // the following code does: sum += A[i][k] * B[k][j];

      aPart = _mm_loadu_ps(&(A[arguments->i][k].real));
      aPart = _mm_shuffle_ps(aPart,aPart, _MM_SHUFFLE(1,0,1,0));
      _mm_store_ps(dest,aPart);
      //printf("A: \n 0: %f\n 1: %f\n 2: %f\n 3: %f\n", dest[0],dest[1],dest[2],dest[3]);

      bPart = _mm_loadu_ps(&B[k][j].real);
      bPart = _mm_shuffle_ps(bPart,bPart, _MM_SHUFFLE(0,1,1,0));
      _mm_store_ps(dest,bPart);
      //printf("B: \n 0: %f\n 1: %f\n 2: %f\n 3: %f\n", dest[0],dest[1],dest[2],dest[3]);


      res = _mm_mul_ps(aPart,bPart);

      _mm_store_ps(dest,res);

      //printf("Res: \n 0: %f\n 1: %f\n 2: %f\n 3: %f\n", dest[0],dest[1],dest[2],dest[3]);



      sum.real += (dest[0] - dest[1]);
      //printf("Sum.real: %f\n",sum.real);
      sum.imag += (dest[2] + dest[3]);
      //printf("Sum.imag: %f\n",sum.imag);
    }
    arguments->C[arguments->i][j] = sum;
  }

  //free(dest);
  pthread_exit(NULL);
}

void fastmul2(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2) {
  int i, j, k;
  float *dest = malloc(sizeof(float)*4);
  __m128 aPart;
  __m128 bPart;
  __m128 res;


  struct complex temp;
  for(i =0; i < a_dim2; i++) {
    for(j =0; j < i; j++) {
      temp.real = B[i][j].real;
      temp.imag = B[i][j].imag;
      B[i][j].real = B[j][i].real;
      B[i][j].imag = B[j][i].imag;
      B[j][i].real = temp.real;
      B[j][i].imag = temp.imag;
    }
  }






  for ( i = 0; i < a_dim1; i++ ) {
    for( j = 0; j < b_dim2; j++ ) {
      struct complex sum;
      sum.real = 0.0;
      sum.imag = 0.0;
      for ( k = 0; k < a_dim2; k++ ) {
          // the following code does: sum += A[i][k] * B[k][j];
          aPart = _mm_loadu_ps(&A[j][k].real);
          bPart = _mm_loadu_ps(&B[i][k].real);
          aPart = _mm_shuffle_ps(aPart,aPart, _MM_SHUFFLE(3,2,3,2));
          bPart = _mm_shuffle_ps(bPart,bPart, _MM_SHUFFLE(3,2,2,3));
          res = _mm_mul_ps(aPart,bPart);

          _mm_store_ps(dest,res);
          sum.real += dest[0] - dest[1];
          sum.imag += dest[2] + dest[3];
      }
      C[i][j] = sum;
    }
  }
  write_out(C,2,2);

  for(i =0; i < b_dim2; i++) {
    for(j =0; j < i; j++) {
      temp.real = C[i][j].real;
      temp.imag = C[i][j].imag;
      C[i][j].real = B[j][i].real;
      C[i][j].imag = B[j][i].imag;
      C[j][i].real = temp.real;
      C[j][i].imag = temp.imag;
    }
  }
}


void fastmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2)
{
  int numberOfThreads = a_dim1;
  printf("Number of threads: %d\n",numberOfThreads);
  pthread_t threads[numberOfThreads];
  struct arguments ** threadArguments = malloc(sizeof(struct arguments) *  numberOfThreads);
  int threadArgIndex = 0;
  int i, j, k;

  printf("Staring threads\n");
  for ( i = 0; i < a_dim1; i++ ) {
    threadArguments[threadArgIndex] = new_arguments(A,B,C,a_dim2,b_dim2,i);
    pthread_create(&threads[threadArgIndex], NULL, calculate_element, (void *)threadArguments[threadArgIndex]);
    threadArgIndex++;
  }
  printf("Finished starting threads\n");

  for(i=0; i < numberOfThreads; i++) {
    pthread_join(threads[i], NULL);
  }
}

void * calculate_element2(void * args) {
  struct arguments2 * arguments = (struct arguments2 *) args;
  struct complex sum;
  sum.real = 0.0;
  sum.imag = 0.0;
  float *dest = malloc(sizeof(float)*4);
  int k,j;
  __m128 aPart;
  __m128 bPart;
  __m128 res;
  struct complex product;
  product.real = 0.0;
  product.imag = 0.0;
  int i;
  for(i=arguments->start_i; i < arguments->end_i; i++) {
    for(j = 0; j < arguments->max_j; j++) {
      struct complex product;
      sum.real = 0.0;
      sum.imag = 0.0;
      for ( k = 0; k < arguments->max_k; k++ ) {
        // the following code does: sum += A[i][k] * B[k][j];

        aPart = _mm_loadu_ps(&(A[i][k].real));
        aPart = _mm_shuffle_ps(aPart,aPart, _MM_SHUFFLE(1,0,1,0));
        _mm_store_ps(dest,aPart);
        //printf("A: \n 0: %f\n 1: %f\n 2: %f\n 3: %f\n", dest[0],dest[1],dest[2],dest[3]);

        bPart = _mm_loadu_ps(&B[k][j].real);
        bPart = _mm_shuffle_ps(bPart,bPart, _MM_SHUFFLE(0,1,1,0));
        _mm_store_ps(dest,bPart);
        //printf("B: \n 0: %f\n 1: %f\n 2: %f\n 3: %f\n", dest[0],dest[1],dest[2],dest[3]);


        res = _mm_mul_ps(aPart,bPart);

        _mm_store_ps(dest,res);

        //printf("Res: \n 0: %f\n 1: %f\n 2: %f\n 3: %f\n", dest[0],dest[1],dest[2],dest[3]);



        sum.real += (dest[0] - dest[1]);
        //printf("Sum.real: %f\n",sum.real);
        sum.imag += (dest[2] + dest[3]);
        //printf("Sum.imag: %f\n",sum.imag);
      }
      arguments->C[i][j] = sum;
    }
  }

  //free(dest);
  pthread_exit(NULL);
}

void fastmul3(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2) {
  int numberOfThreads = 32;
  pthread_t threads[numberOfThreads];
  struct arguments2 ** threadArguments = malloc(sizeof(struct arguments2) * numberOfThreads);
  int count = 0;
  int threadArgIndex = 0;
  int increment = a_dim1 / numberOfThreads;
  int nextValue;

  printf("Staring threads\n");
  while(count < a_dim1) {
    if(count + increment >= a_dim1) {
      nextValue = a_dim1-1;
    } else {
      nextValue = count+increment;
    }
    threadArguments[threadArgIndex] = new_arguments2(A,B,C,a_dim2,b_dim2,count, nextValue);
    pthread_create(&threads[threadArgIndex], NULL, calculate_element2, (void *)threadArguments[threadArgIndex]);
    threadArgIndex++;

    if(nextValue == a_dim1-1) {
      count = a_dim1;
    } else {
      count = nextValue;
    }
    printf("Count: %d\n",count);
  }
  printf("Finished starting threads\n");

  int i;
  for(i=0; i < numberOfThreads; i++) {
    pthread_join(threads[i], NULL);
  }

}



/* write matrix to stdout */
void write_out(struct complex ** a, int dim1, int dim2)
{
  int i, j;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2 - 1; j++ ) {
      printf("%f + %fi ", a[i][j].real, a[i][j].imag);
    }
    printf("%f +%fi\n", a[i][dim2-1].real, a[i][dim2-1].imag);
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

  struct complex ** result;
  int i, j;
  struct timeval seedtime;
  int seed;

  result = new_empty_matrix(dim1, dim2);

  // use the microsecond part of the current time as a pseudorandom seed
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  // fill the matrix with random numbers
  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      long long upper = random();
      long long lower = random();
      result[i][j].real = (float)((upper << 32) | lower);
      upper = random();
      lower = random();
      result[i][j].imag = (float)((upper << 32) | lower);
    }
  }


  /*
  int index = 1;
  for( i=0; i < dim1; i++) {
    for( j=0; j < dim2; j++) {
      result[i][j].real = index;
      index++;
      result[i][j].imag = index;
      index++;
    }
  }
  */

  return result;
}

/* check the sum of absolute differences is within reasonable epsilon */
void check_result(struct complex ** result, struct complex ** control, int dim1, int dim2)
{
  int i, j;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      double diff;
      diff = fabsf(control[i][j].real - result[i][j].real);
      sum_abs_diff = sum_abs_diff + diff;
      diff = fabsf(control[i][j].imag - result[i][j].imag);
      sum_abs_diff = sum_abs_diff + diff;
    }
  }
  //sum_abs_diff = fabs(sum_abs_diff);
  if ( sum_abs_diff > EPSILON ) {
    printf("WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",sum_abs_diff, EPSILON);
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

/* the fast version of matmul written by the team */
void team_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2)
{
  // this call here is just dummy code
  // insert your own code instead
  fastmul2(A, B, C, a_dim1, a_dim2, b_dim2);
  //matmul(A, B, C, a_dim1, a_dim2, b_dim2);
}

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

struct complex ** fastTrans(struct complex** A, int dim1, int dim2)
{
	// Minimum 2 rows, 4 columns! i.e 2x4 matrix
	struct complex** res = new_empty_matrix(dim2, dim1);
	int i, j;
	//NO idea if faster? O.o 
	#pragma omp parallel for
	for(i = 0; i<dim1; i+=2)
	{
		for(j = 0; j<dim2; j+=4)
		{
			int t = (j + 4)>dim2? dim2 - 4: j;
			int t2 = (i+2)>dim1? dim1 - 2: i;
			trans(A,res, t2, t, dim1); 
		}
	}
	write_out(res, dim2, dim1);
}

int main(int argc, char ** argv)
{
  struct complex ** C;
  struct complex ** control_matrix;
  long long mul_time;
  int a_dim1, a_dim2, b_dim1, b_dim2;
  struct timeval start_time;
  struct timeval stop_time;

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



  DEBUGGING(write_out(A, a_dim1, a_dim2));

  //use a simple matmul routine to produce control result
  matmul(A, B, control_matrix, a_dim1, a_dim2, b_dim2);
  //fastmul(A, B, control_matrix, a_dim1, a_dim2, b_dim2);

  // record starting time
  gettimeofday(&start_time, NULL);

  // perform matrix multiplication

  fastmul3(A, B, C, a_dim1, a_dim2, b_dim2);

  // record finishing time
  gettimeofday(&stop_time, NULL);
  mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Matmul time: %lld microseconds\n", mul_time);

  DEBUGGING(write_out(C, a_dim1, b_dim2));

  // now check that the team's matmul routine gives the same answer
  // as the known working version
  check_result(C, control_matrix, a_dim1, b_dim2);

  return 0;
}
