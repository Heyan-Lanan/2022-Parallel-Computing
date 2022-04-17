#include <cstdio>
#include <chrono>
#include <iostream>
#include "C:\Program Files\JetBrains\CLion 2021.3.3\bin\mingw\lib\gcc\x86_64-w64-mingw32\11.2.0\include\omp.h"
#include <C:\MinGW\include\gmp.h>
#include <C:\MinGW\include\mpfr.h>
#define NUM 2;

using namespace std;

int main ()
{
    unsigned int i;
    mpfr_t s, t[200], u;
    int N = 500;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(int k = 1; k <= 101; k++)
        mpfr_init2 (t[k], N);
    mpfr_set_d (t[1], 1.0, MPFR_RNDD);
    mpfr_init2 (s, N);
    mpfr_set_d (s, 0.0, MPFR_RNDD);
    mpfr_init2 (u, N);

    #pragma omp parallel for
    for (i = 1; i <= 100; i++)
    {
        mpfr_set_d (u, double(i), MPFR_RNDD);
        mpfr_div (t[i+1], t[i], u, MPFR_RNDD);
    }
    for (i = 1; i <= 100; i++){
        mpfr_add (s, s, t[i], MPFR_RNDD);
    }

    printf ("Sum is ");
    mpfr_out_str (stdout, 10, 0, s, MPFR_RNDD);
    putchar ('\n');
    mpfr_clear (s);
    for(int k = 1; k <= 100; k++)
        mpfr_clear (t[k]);
    mpfr_clear (u);
    mpfr_free_cache ();
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    cout << "Timing: " << chrono::duration_cast<chrono::duration<double>>(t2 - t1).count() << "s";
    return 0;
}