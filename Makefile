PCNT =
DEFS =

CC := gcc

CFLAGS   := -O0 -Wall -Werror -fno-common $(DEFS)
CPPFLAGS := -DXXX -I./include

LDFLAGS  :=
LDLIBS   := -lm -lcrypto

TARGET := test
SRCS := src/main.c src/nn_subgadget.c src/nn_gadget.c src/unmasked.c
HDRS := include/main.h include/nn_subgadget.h include/nn_gadget.h include/unmasked.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SRCS) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)
