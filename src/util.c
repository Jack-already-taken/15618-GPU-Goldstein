/*
 * util.c -- utility Functions: I/O, memory, etc.
 */
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include "pi.h"
#include "file.h"
#include "util.h"

/* Print error message and exit */
void ErrorHandler(const char *msg, const char *item, int code)
{
  fprintf(stderr, "%s\nItem: %s\n", msg, item);
  exit(code);
}

/* open the file */
void OpenFile(FILE **fp, char *name, const char *mode)
{
  if ((*fp = fopen(name, mode))==NULL)
    ErrorHandler("Cannot open file", (const char *)name, FILE_OPEN_ERROR);
}

/* Search for the keyword in the string. If present, return 1. */
int Keyword(char *string, char *keyword)
{
  int  i, lenstr, lenkey;
  char str[500], key[500];
  lenstr = strlen(string);
  lenkey = strlen(keyword);
  if (lenstr != lenkey) return 0;
  str[lenstr] = key[lenstr] = 0;
  for (i=0; i<=lenstr; i++)
    str[i] = tolower(string[i]);
  for (i=0; i<=lenstr; i++)
    key[i] = tolower(keyword[i]);
  return (!strcmp(str, key));
}

/* Allocate array of bytes and initialize to zero. */
void AllocateByte(unsigned char **ptr, int len, const char *name)
{
  int i;
  *ptr = (unsigned char *) malloc(len*sizeof(unsigned char));
  if ((*ptr)==NULL)
    ErrorHandler("Cannot allocate memory", name,
                 MEMORY_ALLOCATION_ERROR);
  for (i=0; i<len; i++) (*ptr)[i] = 0;
}

/* Allocate array of short integers  and initialize to zero. */
void AllocateShort(short **ptr, int len, const char *name)
{
  int i;
  *ptr = (short *) malloc(len*sizeof(short));
  if ((*ptr)==NULL)
    ErrorHandler("Cannot allocate memory", name,
                 MEMORY_ALLOCATION_ERROR);
  for (i=0; i<len; i++) (*ptr)[i] = 0;
}

/* Allocate array of integers and initialize to zero. */
void AllocateInt(int **ptr, int len, const char *name)
{
  int i;
  *ptr = (int *) malloc(len*sizeof(int));
  if ((*ptr)==NULL)
    ErrorHandler("Cannot allocate memory", name,
                 MEMORY_ALLOCATION_ERROR);
  for (i=0; i<len; i++) (*ptr)[i] = 0;
}

/* Allocate array of floats and initialize to zero. */
void AllocateFloat(float **ptr, int len, const char *name)
{
  int i;
  *ptr = (float *) malloc(len*sizeof(float));
  if ((*ptr)==NULL)
    ErrorHandler("Cannot allocate memory", name,
                 MEMORY_ALLOCATION_ERROR);
  for (i=0; i<len; i++) (*ptr)[i] = 0.0;
}

/* Allocate array of doubles and initialize to zero. */
void AllocateDouble(double **ptr, int len, const char *name)
{
  int i;
  *ptr = (double *) malloc(len*sizeof(double));
  if ((*ptr)==NULL)
    ErrorHandler("Cannot allocate memory", name,
                 MEMORY_ALLOCATION_ERROR);
  for (i=0; i<len; i++) (*ptr)[i] = 0.0;
}

/* Read array of bytes */
void ReadByte(FILE *fp, unsigned char *data, int len, char *name)
{
  if (len != fread(data, sizeof(unsigned char), len, fp))
    ErrorHandler("File read error", name, FILE_READ_ERROR);
  fclose(fp);
}

/* Read array of short integers */
void ReadShort(FILE *fp, short *data, int len, char *name)
{
  if (len != fread(data, sizeof(short), len, fp))
    ErrorHandler("File read error", name, FILE_READ_ERROR);
  fclose(fp);
}

/* Read array of integers */
void ReadInt(FILE *fp, int *data, int len, char *name)
{
  if (len != fread(data, sizeof(int), len, fp))
    ErrorHandler("File read error", name, FILE_READ_ERROR);
  fclose(fp);
}

/* Read array of floats */
void ReadFloat(FILE *fp, float *data, int len, char *name)
{
  if (len != fread(data, sizeof(float), len, fp))
    ErrorHandler("File read error", name, FILE_READ_ERROR);
  fclose(fp);
}

/* Read array of double-precision floats */
void ReadDouble(FILE *fp, double *data, int len, char *name)
{
  if (len != fread(data, sizeof(double), len, fp))
    ErrorHandler("File read error", name, FILE_READ_ERROR);
  fclose(fp);
}

/* Write array of bytes */
void WriteByte(FILE *fp, unsigned char *data, int len, char *name)
{
  if (len != fwrite(data, sizeof(unsigned char), len, fp))
    ErrorHandler("File write error", name, FILE_WRITE_ERROR);
  fclose(fp);
}

/* Write array of short integers */
void WriteShort(FILE *fp, short *data, int len, char *name)
{
  if (len != fwrite(data, sizeof(short), len, fp))
    ErrorHandler("File write error", name, FILE_WRITE_ERROR);
  fclose(fp);
}

/* Write array of doubles */
void WriteDouble(FILE *fp, double *data, int len, char *name)
{
  if (len != fwrite(data, sizeof(double), len, fp))
    ErrorHandler("File write error", name, FILE_WRITE_ERROR);
  fclose(fp);
}

/* Averages byte values and scale from 0-255 to 0-1.  Store      */
/* results in array "out".  tsize is size of averaging template. */
void AverageByteToFloat(unsigned char *in, float *out, int tsize,
                        int xsize, int ysize)
{
  int  hs, i, j, n, ii, jj, c, sum;
  double  scale=1.0/255.0;
  hs = tsize/2;
  for (j=0; j<ysize; j++) {
    for (i=0; i<xsize; i++) {
      for (n=0, sum=0.0, jj=j-hs; jj<=j+hs; jj++) {
        if (jj < 0 || jj > ysize - 1) continue;
        for (ii=i-hs; ii<=i+hs; ii++) {
          if (ii < 0 || ii > xsize - 1) continue;
          c = in[jj*xsize + ii];
          sum += c;
          ++n;
        }
      }
      out[j*xsize + i] = scale*sum/n;
    }
  }
}

/* Search for the keyword "key" in the command line arguments, */
/* then read the value into the variable pointed to by ptr.    */
/* If required = 1, exit if the parameter is not found.  Print */
/* the string "usage" in this case before exiting.             */
int CommandLineParm(int argc, char *argv[], char *key,
  CommandLineParmType type, void *ptr, int required, char *usage)
{
  int  i, found=0;
  for (i=1; i<argc; i++) {
    if (Keyword(argv[i], key)) {
      if (i < argc - 1) {
        found = 1;
        if (type==IntegerParm)
          sscanf(argv[i+1], "%d", (int *)ptr);
        else if (type==FloatParm)
          sscanf(argv[i+1], "%f", (float *)ptr);
        else if (type==DoubleParm)
          sscanf(argv[i+1], "%lf", (double *)ptr);
        else if (type==StringParm)
          sscanf(argv[i+1], "%s", (char *)ptr);
        break;
      }
      else {
        fprintf(stderr, "Missing parameter value for %s\n",
                argv[i]);
        fprintf(stderr, "%s", usage);
        exit(BAD_USAGE);
      }
    }
  }
  if (required && !found) {
    fprintf(stderr, "Required parameter missing: %s\n", key);
    fprintf(stderr, "%s", usage);
    exit(BAD_USAGE);
  }
  return found;
}

/* Compute and print the min & max of the array "soln". */
void PrintMinAndMax(int w, int h, float *soln, char *keyword)
{
  int   i, j;
  float rmin, rmax;
  printf("Finding min, max of %s...", keyword);
  rmin = rmax = soln[0];
  for (j=0; j<w*h; j++) {
    if (soln[j] > rmax) rmax = soln[j];
    if (soln[j] < rmin) rmin = soln[j];
  }
  printf("Min %f, max %f\n", rmin, rmax);
}
