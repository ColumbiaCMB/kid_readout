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

	int sock, length;
	socklen_t tolen;
	struct sockaddr_in server;
	struct sockaddr_in to;
	struct hostent *hp;
	FILE* myfile;
	if (argc < 3) {
		fprintf(stderr, "Usage: process id, ppout specific\n");
		return 0;
	}
	sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (sock < 0)
		error("ERROR opening socket");
	length = sizeof(server);
	bzero(&server, length);
	to.sin_family = AF_INET;
	hp = gethostbyname("192.168.1.1");
	// Readout IP.
	if (hp == 0)
		error("Unknown host");
	bcopy((char *) hp->h_addr, (char *) &to.sin_addr, hp->h_length);
	to.sin_port = htons(atoi("12345"));
	tolen = sizeof(struct sockaddr_in);
	// This code is from fileserver.c to create connections.
	// Refer to fileserver.c for details on how it works.

	char addr_specs_a[256];
	sprintf(addr_specs_a, "/proc/%s/hw/ioreg/%s_a", argv[1], argv[2]);
	// For testing argv[1] shoudl be 641 and argv[2] should be ppout. Or something like that..

	char my_buffer[1024];
	int ii;
	printf("This program does a sendto 1,000,000 times.\n");
	my_file = fopen(addr_specs_a, "r");
	result = fread(my_buffer, 1, 1024, my_file);
	fclose(my_file);

	int n;

	for (ii = 0; ii < 1000000; ii++) {
		n = sendto(sock, my_buffer, 1024, 0, (struct sockaddr *) &to, tolen);
	}

	return 0;
}
