#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

unsigned int read_int(char *reg_name) {
    // ~278 us, 14400 bytes/sec
    FILE* regfile;
    unsigned int result;
    int rval;

    regfile = fopen(reg_name,"r");
    rval = fread((char *)(&result), 1, 4, regfile);
    fclose(regfile);
    return result;
}

int read_bram(char *bram_name, char *dest, int bytes) {
    // 16384 bytes in 2.35 ms, ~7 MB/s
    FILE* regfile;
    int rval;

    regfile = fopen(bram_name,"r");
    rval = fread(dest, bytes, 1, regfile);
    fclose(regfile);
    return rval;
}

int main(int argc, char *argv[]) {
    int sock;
    socklen_t tolen;
    struct sockaddr_in to;
    struct hostent *hp;
    char pktbuf[1500];
    int rval;

    int i;
    int pn;
    int idle;
    int result;
    char addr_regname[256];
    char banka_regname[256];
    char bankb_regname[256];
    char chansel_regname[256];
    char mcntr_regname[256];
    FILE* bramfile;

    int addr;
    char bank;
    int last_addr;
    char last_bank;
    unsigned int chansel;
    unsigned int mcntr;
    int chan;
    int streamid;
    if (argc < 2){
        return -1;
    }
    sprintf(addr_regname,"/proc/%s/hw/ioreg/ppout0_addr",argv[1]);
    sprintf(banka_regname,"/proc/%s/hw/ioreg/ppout0_a",argv[1]);
    sprintf(bankb_regname,"/proc/%s/hw/ioreg/ppout0_b",argv[1]);
    sprintf(chansel_regname,"/proc/%s/hw/ioreg/ppout0_chan",argv[1]);
    sprintf(mcntr_regname,"/proc/%s/hw/ioreg/mcntr",argv[1]);


    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
        error("ERROR opening socket");
    to.sin_family = AF_INET;
    hp = gethostbyname("192.168.1.1");
    // Readout IP.
    if (hp == 0)
        error("Unknown host");
    bcopy((char *) hp->h_addr, (char *) &to.sin_addr, hp->h_length);
    to.sin_port = htons(atoi("12345"));
    tolen = sizeof(struct sockaddr_in);
/*
    n = sendto(sock, filebuffer, buff_length, 0,
            (struct sockaddr *) &to, tolen);
    if (n < 0)
        error("ERROR sending");
*/

    addr = 0x1FFF & read_int(addr_regname);
    last_bank = addr > 4096;
    bank = last_bank;
    
    while (1) {
        idle = 0;
        while (bank == last_bank) {
            addr = 0x1FFF & read_int(addr_regname);
            bank = addr > 4096;
            idle += 1;
        }
        last_bank = bank;
        chansel = read_int(chansel_regname);
        chan = chansel & 0xFFFF;
        streamid = (chansel>>16);
        if (streamid){
            mcntr = read_int(mcntr_regname);
            if (bank) {
//                printf("reading bank a\n");
                bramfile = fopen(banka_regname,"r");
            }
            else {
//                printf("reading bank b\n");
                bramfile = fopen(bankb_regname,"r");
            }
            for(pn = 0; pn < 16; pn++) {
                ((unsigned int *)pktbuf)[0] = (idle<<16) + pn;
                ((unsigned int *)pktbuf)[1] = chansel;
                ((unsigned int *)pktbuf)[2] = mcntr;
                fread(pktbuf+4*3, 1024, 1, bramfile);
                rval = sendto(sock, pktbuf, 1024+4*3, 0,
                        (struct sockaddr *) &to, tolen);
                if (rval < 0)
                    error("ERROR sending");
            }
            fclose(bramfile);
        }
    }
}