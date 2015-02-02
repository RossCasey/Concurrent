#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>


pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
long double totalSum = 0;

struct arguments {
		int startingDenominator;
		int numberOfThreads;
		int numberOfTerms;
		long double * sumLocation;
};

/*
* new_arguments() function:
* This function returns a pointer to a newly created struct arguments in memory. The struct
* arguments has been initialised with the values passed into this function.
*/
struct arguments * new_arguments(int startingDenominator, int numberOfThreads, int numberOfTerms) {
		struct arguments * newArgs = malloc(sizeof(struct arguments));
		newArgs->numberOfTerms = numberOfTerms;
		newArgs->numberOfThreads = numberOfThreads;
		newArgs->startingDenominator = startingDenominator;
		return newArgs;
}

/*
* calculate_pi() function:
* This function is a thread that computes certain terms in the infinite series that converges to pi.
* The function calculates the following series:
*
* pi = 3 + 4/(2x3x4) - 4/(4x5x6) + 4/(6x7x8) - 4/(8x9x10) etc.
*
* The input dictates the terms that the function calculates. Eg if there are 4 threads then each
* thread will compute every fourth term.
*/
void * calculate_pi(void * args) {
		struct arguments * arguments = (struct arguments *) args;
		int increment = (arguments->numberOfThreads * 2);
		int denom = arguments->startingDenominator;
		long double sum = 0;
		long double numerator = 4;

		for(int i=0; i < arguments->numberOfTerms; i++) {
				long double currentDenominator = (long double) denom * (denom + 1) * (denom + 2);

				//in this series the term is subtracted if the first denominator is divisible by 4.
				if(denom % 4 == 0) {
						sum -= numerator / currentDenominator;
				} else {
						sum += numerator / currentDenominator;
				}
				denom += increment;
		}

		pthread_mutex_lock(&mutex1);            //mutex lock before addition
		totalSum += sum;
		pthread_mutex_unlock(&mutex1);          //unlock
		pthread_exit(NULL);
}



int main(int argc, const char * argv[])
{
		int numberOfThreads;
		int numberOfTerms;
		if(argc < 2) {
				numberOfTerms = 10000;
				numberOfThreads = 4;
				printf("No command-line input, will use default: Threads(4) Terms(10000).\n");
		} else {
				numberOfThreads = atoi(argv[1]);
				numberOfTerms = (atoi(argv[2])) / numberOfThreads;
				printf("User input: Threads(%d) Terms(%d).\n", numberOfThreads, numberOfTerms * numberOfThreads);
		}

		pthread_t threads[numberOfThreads];         //create array of threads

		//create an array of arguments to pass to each thread
		struct arguments ** threadArguments = malloc(sizeof(struct arguments) *  numberOfThreads);
		for(int i=0; i < numberOfThreads; i++) {
				threadArguments[i] = new_arguments((i * 2) + 2, numberOfThreads, numberOfTerms);
		}

		//start each thread
		for(int i=0; i < numberOfThreads; i++) {
				pthread_create(&threads[i], NULL, calculate_pi, (void *)threadArguments[i]);
		}

		//join all threads
		for(int i=0; i < numberOfThreads; i++) {
						pthread_join(threads[i], NULL);
		}



		//free malloced memory
		for(int i=0; i < numberOfThreads; i++) {
				free(threadArguments[i]);
		}


		//print calculation
		pthread_mutex_lock(&mutex1);            //mutex lock before addition
		totalSum += 3;
		pthread_mutex_unlock(&mutex1);          //unlock
		printf("Pi Calculation: %.64Lf\n", totalSum);
		pthread_mutex_destroy(&mutex1);

		return 0;

}
