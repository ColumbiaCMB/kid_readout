#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#define FALSE 0
#define TRUE !(FALSE)

int main(int argc, char *argv[]) {
	FILE* my_file;
	int result;

	if (argc < 3) {
		fprintf(stderr, "Usage: process id, ppout specific\n");
		return 0;
	}

	char addr_specs_chan[256];
	sprintf(addr_specs_chan, "/proc/%s/hw/ioreg/%s_chan", argv[1], argv[2]);
	// For testing argv[1] shoudl be 641 and argv[2] should be ppout. Or something like that..

	char my_buffer[1024];
	int i;
	int e;
	for (i = 0; i < 10000; i++) {
		my_file = fopen(addr_specs_chan, "r");
		for (e = 0; e < 16; e++) {
			result = fread(my_buffer, 1, 4, my_file);
		}
		fclose(my_file);
	}
	printf("This program does 10000x16 file reads of addr_specs_chan.");
	// Print description of what the program does.
	return 0;
}
