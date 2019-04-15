#ifndef HAWKES_TYPES_H_
#define HAWKES_TYPES_H_

struct DeviceData {
	float *mu;
	float *k0;
	float *w;
	float *t_series;
	float *intensity_series;
	float *log_likelikood;
	float *diag_p_arr;
	float *p_arr;
	float *p_t_arr;
};

struct HostData {
	float mu;
	float k0;
	float w;
	float *t_series;
	float *intensity_series;
	float log_likelikood;
};

const int DATA_SIZE = 1;
const int DB_SIZE = 25;

const char* OUTPUT = "outputTimes.txt";
const char* FILENAME = "LockeLowell.csv";

#define PI 3.14159265
#define T double

#endif /* HAWKES_TYPES_H_ */
