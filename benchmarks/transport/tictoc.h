#ifndef _TICTOC_H
#define _TICTOC_H

struct timeval tb, te;
double bytes = 0;

void tic(void)
{ 
#pragma omp master
  {
    gettimeofday(&tb, NULL);
    bytes = 0;
  }
}

void toc(void)
{ 
#pragma omp master
  {
    long s,u;
    double tt;
    gettimeofday(&te, NULL);
    s=te.tv_sec-tb.tv_sec;
    u=te.tv_usec-tb.tv_usec;
    tt=((double)s)*1000000+u;
    fprintf(stderr, "time:                  %li.%.6lis\n", (s*1000000+u)/1000000, (s*1000000+u)%1000000);
    fprintf(stderr, "MB/s:                  %.3lf\n", bytes/tt);
  }
}

#endif
