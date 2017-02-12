#ifndef _NETWORK_H_
#define _NETWORK_H_

// ----------------------------------------------------------------
typedef unsigned long long uint64_t;
typedef long long int64_t;

typedef unsigned int uint32_t;
typedef int int32_t;

typedef unsigned short uint16_t;
typedef short int16_t;

typedef unsigned char uint8_t;
// typedef char int8_t;

// ----------------------------------------------------------------
inline double mk_abs( double d ) {

    if ( d > 0 ) return d;

    double t=0;
    while ( d < 0.00000001 ) {
        t+=0.00001;
        d+=0.00001;

    }
    return t;
}

// ----------------------------------------------------------------

#endif