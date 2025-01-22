#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Function to generate random numbers
void generateRandomNumbers(int *arr, int N, int A, int B) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = A + rand() % (B - A + 1);
    }
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}



int partition(int *arr, int low, int high) {
    int pivot_index = low + (high - low)/2;
    int pivot = arr[pivot_index];

    swap(&arr[pivot_index], &arr[high]);

    int current_index = low;

    for (int i = low; i < high; i++) {
        if (arr[i] < pivot) {
            swap(&arr[current_index], &arr[i]);
            current_index++;
        }
    }
    swap(&arr[current_index], &arr[high]);
    return current_index;
}

void quickSort1(int *arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        
        quickSort1(arr, low, pi - 1);

        quickSort1(arr, pi + 1, high);
    }
}

void quickSort2(int *arr, int low, int high, int depth) {
    if (low < high) {
        int pi = partition(arr, low, high);

        depth++;

        if (depth < 10) {
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    quickSort2(arr, low, pi - 1, depth);
                }

                #pragma omp section
                {
                    quickSort2(arr, pi + 1, high, depth);
                }
            }
        }
        else{
            quickSort1(arr, low, pi - 1);

            quickSort1(arr, pi + 1, high);
        }
        
    }
}

// Recursive quicksort function with OpenMP


int main() {
    omp_set_nested(1);
    int N;
    printf("Enter number of elements: \n");
    scanf("%d", &N);

    int *randomNumbers = (int *)malloc(N * sizeof(int));
    if (randomNumbers == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    generateRandomNumbers(randomNumbers, N, 1, 100000);

    // printf("Original array: \n");
    // for (int i = 0; i < N; i++) {
    //     printf("%d ", randomNumbers[i]);
    // }
    // printf("\n");

    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            quickSort2(randomNumbers, 0, N - 1, 0);
        }
    }

    double end_time = omp_get_wtime();

    // printf("\nSorted array: \n");
    // for (int i = 0; i < N; i++) {
    //     printf("%d ", randomNumbers[i]);
    // }
    // printf("\n");
    int diffrence = 0;
    for (int i = 0; i < N - 1; i++) {
        if (randomNumbers[i] > randomNumbers[i + 1]) {
            diffrence++;
            break;
        }
        
    }

    if (diffrence == 1) {
        printf("doesn't works");
    }
    else {
        printf("works");
    }

    printf("Time taken: %f seconds\n", end_time - start_time);

    free(randomNumbers);
    return 0;
}
