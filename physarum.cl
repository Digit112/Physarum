// Test kernel that has each work-item print its global ID.
kernel void hullo() {
	printf("hullo ! (%lu)\n", get_global_id(0));
}

// Apply a box blur to the passed image. diffK is the width of the kernel and must be odd. DecayT is the percent of chemoattractant removed from each cell.
kernel void blur(int width, int height, int diffK, float decayT, global ushort* from, global ushort* to) {
	int areaK = diffK*diffK;
	diffK /= 2;

	int Mx = get_global_id(0) + diffK;
	int My = get_global_id(1) + diffK;

	uint tot = 0;
	for (int x = get_global_id(0) - diffK; x <= Mx; x++) {
		for (int y = get_global_id(1) - diffK; y <= My; y++) {
			size_t ind = ((x + width) % width) + ((y + height) % height) * width;
			tot += (uint) from[ind];
		}
	}

	to[get_global_id(0) + get_global_id(1) * width] = (ushort) ((2*tot + areaK) / (areaK*2) * (1 - decayT) + 0.5);
}
