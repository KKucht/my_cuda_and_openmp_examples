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

// Recursive quicksort function with OpenMP
void quickSort(int *arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                quickSort(arr, low, pi - 1);
            }

            #pragma omp section
            {
                quickSort(arr, pi + 1, high);
            }
        }
    }
}

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
            quickSort(randomNumbers, 0, N - 1);
        }
    }

    double end_time = omp_get_wtime();

    // printf("\nSorted array: \n");
    // for (int i = 0; i < N; i++) {
    //     printf("%d ", randomNumbers[i]);
    // }
    // printf("\n");

    printf("Time taken: %f seconds\n", end_time - start_time);

    free(randomNumbers);
    return 0;
}
