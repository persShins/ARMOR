#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#define TEST_IMAGE  "./data/t10k-images.idx3-ubyte"
#define TEST_LABEL  "./data/t10k-labels.idx1-ubyte"

#define SIZE 784 // 28*28
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1

int info_image[LEN_INFO_IMAGE];
int info_label[LEN_INFO_LABEL];

unsigned char image[MAX_NUM_OF_IMAGES][MAX_IMAGESIZE][MAX_IMAGESIZE];

unsigned char test_image_char[NUM_TEST][SIZE];

unsigned char test_label_char[NUM_TEST][1];

double test_image[NUM_TEST][SIZE];

int test_label[NUM_TEST];


/* 4-byte endian conversion */
void FlipLong(unsigned char *ptr)
{
    unsigned char val;

    // Swap 1st and 4th bytes
    val = *ptr;
    *ptr = *(ptr + 3);
    *(ptr + 3) = val;

    // Swap 2nd and 3rd bytes
    ptr += 1;
    val = *ptr;
    *ptr = *(ptr + 1);
    *(ptr + 1) = val;
}


/* Read only header information */
void read_mnist_info(const char *file_path, int len_info, int info_arr[])
{
    int i, fd;
    ssize_t bytes_read;
    unsigned char *ptr;

    fd = open(file_path, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "couldn't open file: %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    bytes_read = read(fd, info_arr, len_info * sizeof(int));
    if (bytes_read != (ssize_t)(len_info * sizeof(int))) {
        fprintf(stderr, "couldn't read header from file: %s\n", file_path);
        close(fd);
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < len_info; i++) {
        ptr = (unsigned char *)&info_arr[i];
        FlipLong(ptr);
    }

    close(fd);
}


/* Read raw image/label data after header */
void read_mnist_data(const char *file_path, int header_size, int num_data, int arr_n,
                     unsigned char data_char[][arr_n])
{
    int i, fd;
    ssize_t bytes_read;

    fd = open(file_path, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "couldn't open file: %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    if (lseek(fd, header_size, SEEK_SET) == -1) {
        fprintf(stderr, "lseek failed for file: %s\n", file_path);
        close(fd);
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < num_data; i++) {
        bytes_read = read(fd, data_char[i], arr_n * sizeof(unsigned char));
        if (bytes_read != (ssize_t)(arr_n * sizeof(unsigned char))) {
            fprintf(stderr, "couldn't read data[%d] from file: %s\n", i, file_path);
            close(fd);
            exit(EXIT_FAILURE);
        }
    }

    close(fd);
}

/* Convert unsigned char image data to double */
void image_char2double(int num_data, unsigned char data_image_char[][SIZE], double data_image[][SIZE])
{
    int i, j;
    for (i = 0; i < num_data; i++) {
        for (j = 0; j < SIZE; j++) {
            data_image[i][j] = (double)data_image_char[i][j] / 255.0;
        }
    }
}


/* Convert unsigned char label data to int */
void label_char2int(int num_data, unsigned char data_label_char[][1], int data_label[])
{
    int i;
    for (i = 0; i < num_data; i++) {
        data_label[i] = (int)data_label_char[i][0];
    }
}


/* Load image file only */
void load_mnist_images(const char *image_path, int num_data,
                       unsigned char image_char[][SIZE],
                       double image[][SIZE])
{
    read_mnist_info(image_path, LEN_INFO_IMAGE, info_image);
    read_mnist_data(image_path, LEN_INFO_IMAGE * sizeof(int), num_data, SIZE, image_char);
    image_char2double(num_data, image_char, image);
}

/* Load label file only */
void load_mnist_labels(const char *label_path, int num_data,
                       unsigned char label_char[][1],
                       int label[])
{
    read_mnist_info(label_path, LEN_INFO_LABEL, info_label);
    read_mnist_data(label_path, LEN_INFO_LABEL * sizeof(int), num_data, 1, label_char);
    label_char2int(num_data, label_char, label);
}

/* Load only test dataset */
void load_mnist_test(void)
{
    load_mnist_images(TEST_IMAGE, NUM_TEST, test_image_char, test_image);
    load_mnist_labels(TEST_LABEL, NUM_TEST, test_label_char, test_label);
}