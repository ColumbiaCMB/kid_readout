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

	char addr_specs_a[256];
	sprintf(addr_specs_a, "/proc/%s/hw/ioreg/%s_a", argv[1], argv[2]);
	// For testing argv[1] shoudl be 641 and argv[2] should be ppout. Or something like that..

	char my_buffer[1024];
	int ii;
	printf("This program does a file open of addr_specs_a 100,000 times.\n");
	for (ii = 0; ii < 100000; ii++) {
		my_file = fopen(addr_specs_a, "r");
		result = fread(my_buffer, 1, 1024, my_file);
		fclose(my_file);
	}
	return 0;
}
