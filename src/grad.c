/*
 *   grad.c -- function for computing phase derivative 
 *             (wrapped phase difference)
 */
#include <stdio.h>
#include <math.h>
#include "grad.h"
#include "pi.h"

/* Wrapped phase difference (radians); p1,p2 principal in ~[-PI, PI]. */
float Gradient(float p1, float p2)
{
  float r = p1 - p2;
  if (r > (float)PI)
    r -= (float)TWOPI;
  else if (r < -(float)PI)
    r += (float)TWOPI;
  return r;
}
