#include <stdint.h>
#include <stdio.h>

#include "physarum_init.hpp"

int main() {
	printf("Hello Physarum!\n");

	int f_skip = 2;

	int width = 1920;
	int height = 1080;
	float pop_dens = 0.015;

	physarum2 phy(width, height, pop_dens);
	phy.initialize(0);

	phy.decayT = 0.1;
	phy.RA = M_PI / 8;
	phy.SO = 9;
	phy.SS = 1.5;
	phy.SW = 3;

	phy.update_args();

	printf("%d agents.\n", phy.agents_n);
	uint8_t* img = (uint8_t*) malloc(width*height);
	char buf[128];
	float a = 20000;

	for (int i = 0; i < 1600; i++) {
		printf("Frame %d...\n", i);

		if (i % f_skip == 0) {
			for (int i = 0; i < width*height; i++) {
				img[i] = (uint8_t) (255 - a / (phy.l_deposit[i] + a/255));
	//			img[i] = phy.occupancy[i] * 255;
			}

			sprintf(buf, "out/%04d.pgm", i/f_skip);
			FILE* fout = fopen(buf, "wb");

			int hdr_len = sprintf(buf, "P5 %d %d 255 ", width, height);

			fwrite(buf, 1, hdr_len, fout);
			fwrite(img, 1, width*height, fout);
			fclose(fout);
		}

		phy.step_jones_serial();
	}

	phy.read_deposit();	

	free(img);
		
	return 0;
}
