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

void fileRead(FILE *file_in, int socket_in, struct sockaddr_in destination,
		socklen_t destlen, char* timestamp, char* channel_dir) {

	/*
	 * Description of the variables input into fileRead. Last is the name of the variables
	 * when run by main.
	 *
	 * file_in - the file to be read - myfile
	 * socket_in - the socket - sock
	 * destination - a socket structure describing destination - to
	 * destlen - size of the sock structure - tolen
	 * timestamp - the addr timestamp from the ROACH - addr_buff
	 * channel_dir - the channel stamp for myfile. - addr_specs_chan
	 */

	int buff_length;
	buff_length = 1032;
	char filebuffer[buff_length];
	char channelbuffer[4];
	long loadup;
	int result;
	loadup = 1024;
	int i = 0;
	int n;
	FILE* channel_file;
	int ishift;
	char ishift_or_cbuff[4];

	memset(filebuffer, 0, buff_length);
	/* memset is probably not necessary in the end use.
	 It sets everything to zero, which makes it appear better as a string.
	 Also possible is just setting the fourth value to zero.
	 It is also the only reason string.h has to be included.*/

	memcpy(&filebuffer[4], timestamp, 4);
	// Adds the tick of the ppout_addr so we can arrange files down the road.
	fseek(file_in, 0, SEEK_SET);

	channel_file = fopen(channel_dir, "r");
	result = fread(channelbuffer, 1, 4, channel_file);
	fclose(channel_file);
	// Checks the channel file for a channel, and reads it.

	while (i < 16) {

		ishift = i << 16;
		ishift = ishift | *((int*) channelbuffer);

		// This byte masks i and the channelbuffer to the first four bytes of charbuffer.

		memcpy(&filebuffer, (char*) &ishift, 4);
		// Copies i, again so we can arrange files down the road.
		// Copies to filebuffer as a char array.

		result = fread(filebuffer + 8, 1, loadup, file_in);
		// Reads into buffer from position 8 to end.

		n = sendto(socket_in, filebuffer, buff_length, 0,
				(struct sockaddr *) &destination, destlen);
		if (n < 0)
			error("ERROR sending");
		i++;
	}
}

int main(int argc, char *argv[]) {
	FILE* switch_file;
	char addr_buff[4];
	int result;
	unsigned int addr;
	unsigned int last_addr;
	last_addr = 0;
	// Used for checking discrepancy between subsequent packages.
	long size;
	size = 4;
	//Necessary for the switch file.

	int myswitch;
	int wait;
	wait = TRUE;
	int originalnum;
	int has_switched;
	has_switched = 0;
	// These are to make sure the program reads the file that is not being written to.

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

	char addr_specs_switch[256];
	char addr_specs_a[256];
	char addr_specs_b[256];
	char addr_specs_chan[256];
	sprintf(addr_specs_switch, "/proc/%s/hw/ioreg/%s_addr", argv[1], argv[2]);
	sprintf(addr_specs_a, "/proc/%s/hw/ioreg/%s_a", argv[1], argv[2]);
	sprintf(addr_specs_b, "/proc/%s/hw/ioreg/%s_b", argv[1], argv[2]);
	sprintf(addr_specs_chan, "/proc/%s/hw/ioreg/%s_chan", argv[1], argv[2]);
	// For testing argv[1] should be along the lines of 641 and argv[2] should be ppout.
	// Sets the files from which the program will read.
	// When the program is run without channels, ppout_chan will not exist and you will get a segfault.

	while (1) {
		switch_file = fopen(addr_specs_switch, "r");

		result = fread(addr_buff, 1, size, switch_file);

		addr = *(unsigned int *) addr_buff;
		if (((int) addr) - ((int) last_addr) > 4096) {
			printf("discrepancy between subsequent addrs ");
			printf("%d\n", ((int) addr) - ((int) last_addr));
			// Used for checking discrepancies.
		}
		last_addr = addr;

		myswitch = addr % 8192;

		fclose(switch_file);

		if (wait == TRUE) {
			// Waits for the ROACH to finish writing to read.
			if (has_switched == 0) {
				originalnum = myswitch;
				has_switched = 1;
			}
			if (myswitch < 4096) {
				if (originalnum >= 4096) {
					wait = FALSE;
				}
			}
			if (myswitch >= 4096) {
				if (originalnum < 4096) {
					wait = FALSE;
				}
			}
		} else {
			// Reads from a or b depending on the addr.
			if (myswitch < 4096) {
				myfile = fopen(addr_specs_b, "r");
				fileRead(myfile, sock, to, tolen, addr_buff, addr_specs_chan);
				fclose(myfile);
				wait = TRUE;
				has_switched = 0;
			}
			if (myswitch >= 4096) {
				myfile = fopen(addr_specs_a, "r");
				fileRead(myfile, sock, to, tolen, addr_buff, addr_specs_chan);
				fclose(myfile);
				wait = TRUE;
				has_switched = 0;
			}
		}
	}
	return 0;
}
