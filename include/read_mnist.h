#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>

#define TEST_COMBINED "./data/combined.bytes"

#define WIDTH 28
#define HEIGHT 28
#define SIZE 784
#define NUM_TEST 10000

double test_image[NUM_TEST][SIZE];
int test_label[NUM_TEST];

void read_combined_mnist(const char *file_path,
                         int expected_num,
                         double image[][SIZE],
                         int label[])
{
    int fd;
    int i, j;
    ssize_t bytes_read;

    uint32_t num_images;
    uint32_t rows;
    uint32_t cols;
    uint32_t image_size;

    unsigned char label_buffer;
    unsigned char image_buffer[SIZE];

    fd = open(file_path, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "couldn't open combined file: %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    bytes_read = read(fd, &num_images, sizeof(uint32_t));
    if (bytes_read != (ssize_t)sizeof(uint32_t)) {
        fprintf(stderr, "couldn't read num_images\n");
        close(fd);
        exit(EXIT_FAILURE);
    }

    bytes_read = read(fd, &rows, sizeof(uint32_t));
    if (bytes_read != (ssize_t)sizeof(uint32_t)) {
        fprintf(stderr, "couldn't read rows\n");
        close(fd);
        exit(EXIT_FAILURE);
    }

    bytes_read = read(fd, &cols, sizeof(uint32_t));
    if (bytes_read != (ssize_t)sizeof(uint32_t)) {
        fprintf(stderr, "couldn't read cols\n");
        close(fd);
        exit(EXIT_FAILURE);
    }

    image_size = rows * cols;

    if (num_images != expected_num) {
        fprintf(stderr, "unexpected number of images: %u\n", num_images);
        close(fd);
        exit(EXIT_FAILURE);
    }

    if (rows != WIDTH || cols != HEIGHT) {
        fprintf(stderr, "unexpected image size: %u x %u\n", rows, cols);
        close(fd);
        exit(EXIT_FAILURE);
    }

    if (image_size != SIZE) {
        fprintf(stderr, "unexpected flattened image size: %u\n", image_size);
        close(fd);
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < expected_num; i++) {
        bytes_read = read(fd, &label_buffer, 1);
        if (bytes_read != 1) {
            fprintf(stderr, "couldn't read label[%d]\n", i);
            close(fd);
            exit(EXIT_FAILURE);
        }

        bytes_read = read(fd, image_buffer, SIZE);
        if (bytes_read != SIZE) {
            fprintf(stderr, "couldn't read image[%d]\n", i);
            close(fd);
            exit(EXIT_FAILURE);
        }

        label[i] = (int)label_buffer;

        for (j = 0; j < SIZE; j++) {
            image[i][j] = (double)image_buffer[j] / 255.0;
        }
    }

    close(fd);
}

void load_mnist_test(void)
{
    read_combined_mnist(TEST_COMBINED,
                        NUM_TEST,
                        test_image,
                        test_label);
}
