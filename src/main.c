#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <limits.h>
#include <strings.h>   /* strcasecmp */
#include <getopt.h>
#include <sys/stat.h>
#include <errno.h>
#include "file.h"
#include "util.h"
#include "extract.h"
#include "grad.h"
#include "pi.h"
#include "tiff_io.h"
#include "unwrap_cuda.h"


#define POS_RES     0x01   /* 1st bit */
#define NEG_RES     0x02   /* 2nd bit */
#define VISITED     0x04   /* 3rd bit */
#define ACTIVE      0x08   /* 4th bit */
#define BRANCH_CUT  0x10   /* 5th bit */
#define BORDER      0x20   /* 6th bit */
#define UNWRAPPED   0x40   /* 7th bit */
#define POSTPONED   0x80   /* 8th bit */
#define RESIDUE     (POS_RES | NEG_RES)
#define AVOID       (BRANCH_CUT | BORDER)

int NUM_CORES;


/* mkdir -p */
static int mkdir_p(const char *path)
{
    char tmp[PATH_MAX];
    size_t len;
    char *p;

    if (!path || !*path)
        return -1;
    strncpy(tmp, path, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    len = strlen(tmp);
    while (len > 1 && tmp[len - 1] == '/')
        tmp[--len] = '\0';

    for (p = tmp + 1; *p; p++) {
        if (*p != '/')
            continue;
        *p = '\0';
        if (tmp[0] && mkdir(tmp, 0755) != 0 && errno != EEXIST)
            return -1;
        *p = '/';
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST)
        return -1;
    return 0;
}

static int path_is_existing_dir(const char *path)
{
    struct stat st;
    if (!path || stat(path, &st) != 0)
        return 0;
    return S_ISDIR(st.st_mode);
}

/* basename without extension, ASCII path assumed */
static void input_path_stem(const char *input_path, char *stem, size_t stem_sz)
{
    const char *slash = strrchr(input_path, '/');
    const char *base = slash ? slash + 1 : input_path;

    strncpy(stem, base, stem_sz - 1);
    stem[stem_sz - 1] = '\0';
    {
        char *dot = strrchr(stem, '.');
        if (dot)
            *dot = '\0';
    }
}

double timediff(clock_t t1, clock_t t2) {
    double elapsed;
    elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
    return elapsed;
}

/*
    Exchange two values
 */
void swap(int *a, int *b)
{
    int tmp;

    tmp = *a;
    *a = *b;
    *b = tmp;
}



/*
 *   Computute the x and y gradient.
 */
void Gradxy(float *phase, float *gradx, float *grady, int xsize, int ysize)
{
	int i, j, w;

	#pragma omp parallel for default(none) \
	private(j, i, w) \
	shared(xsize, ysize, phase, gradx, grady)
	for (j=0; j<ysize-1; j++)
	{
		for (i=0; i<xsize-1; i++)
		{
			w = j*xsize + i;
			gradx[w] = Gradient(phase[w], phase[w+1]);
			grady[w] = Gradient(phase[w], phase[w+xsize]);
		}
	}
}




/*
     Returns 0 if no pixels left, 1 otherwise
 */
int GetNextOneToUnwrap(int *a,
                       int *b,
                       int *index_list,
                       int *num_index,
                       int xsize,
                       int ysize)
{
    int index;
    if (*num_index < 1)
        return 0;   /* return if list empty */

    index = index_list[*num_index - 1];
    *a = index%xsize;
    *b = index/xsize;
    --(*num_index);
    return 1;
}




/*
 Insert new pixel into the list.
 Note: qual_map can be NULL
 */
void InsertList(float *soln,
                float val,
                unsigned char *bitflags,
                int a,
                int b,
                int *index_list,
                int *num_index,
                int xsize)
{
    int index;

    index = b*xsize + a;

    soln[index] = val;

    /* add to list */
    index_list[*num_index] = index;

    ++(*num_index);

    bitflags[index] |= UNWRAPPED;

    return;
}




/*
 Insert the four neighboring pixels of the given pixel
 (x,y) into the list.  The quality value of the given
 pixel is "val".
 */
void UpdateList(int x,
                int y,
                float val,
                float *phase,
                float *soln,
                unsigned char *bitflags,
                int xsize,
                int ysize,
                int *index_list,
                int *num_index)
{
    int    i, a, b, k, w;
    float  grad;

    a = x - 1;
    b = y;
    k = b*xsize + a;
    if (a >= 0
        && !(bitflags[k] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
        w = y*xsize + x-1;
        grad = Gradient(phase[w], phase[w+1]);

        InsertList(soln, val + grad, bitflags, a, b,
                   index_list, num_index, xsize);
    }

    a = x + 1;
    b = y;
    k = b*xsize + a;
    if (a < xsize
        && !(bitflags[k] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
        w = y*xsize + x;
        grad = - Gradient(phase[w], phase[w+1]);

        InsertList(soln, val + grad, bitflags, a, b,
                   index_list, num_index, xsize);
    }

    a = x;
    b = y - 1;
    k = b*xsize + a;
    if (b >= 0
        && !(bitflags[k] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
        w = (y-1)*xsize + x;
        grad = Gradient(phase[w], phase[w+xsize]);

        InsertList(soln, val + grad, bitflags, a, b,
                   index_list, num_index, xsize);
    }

    a = x;
    b = y + 1;
    k = b*xsize + a;
    if (b < ysize
        && !(bitflags[k] & (BRANCH_CUT | UNWRAPPED | BORDER))) {
        w = y*xsize + x;
        grad = - Gradient(phase[w], phase[w+xsize]);

        InsertList(soln, val + grad, bitflags, a, b,
                   index_list, num_index, xsize);
    }
}



/* Unwrap the phase data (by Itoh's method) without crossing
 * any branch cuts.  Return number of disconnected pieces.
 */
int UnwrapAroundCutsGoldstein(float *phase,
                     unsigned char *bitflags,
                     float *soln,
                     int xsize,
                     int ysize,
                     int *path_order)
{
    int    i, j, k, a, b, c, n, num_pieces=0;
    float  value;
    int    num_index, max_list_size;
    int    *index_list;

    max_list_size = xsize*ysize;
    AllocateInt(&index_list, max_list_size + 1, "bookkeeping list (index)");


    /* find starting point */
    n = 0;
    num_index = 0;

    for (j=0; j<ysize; j++)
    {
        for (i=0; i<xsize; i++)
        {
            k = j*xsize + i;
            if (!(bitflags[k] & (BRANCH_CUT | UNWRAPPED | BORDER)))
            {
                bitflags[k] |= UNWRAPPED;
                if (bitflags[k] & POSTPONED) /* soln[k] already stores the unwrapped value */
                    value = soln[k];
                else
                {
                    ++num_pieces;
                    value = soln[k] = phase[k];
                }

                UpdateList(i, j, value, phase, soln, bitflags, xsize,
                           ysize, index_list, &num_index);

                while (num_index > 0)
                {
                    ++n;

                    if (!GetNextOneToUnwrap(&a, &b, index_list,
                                            &num_index, xsize, ysize))
                        break;   /* no more to unwrap */

                    c = b*xsize + a;

                    // * * * * * * * * * * *
                    //   Save path order
                    //
                    path_order[c] = n;

                    bitflags[c] |= UNWRAPPED;
                    value = soln[c];
                    UpdateList(a, b, value, phase, soln, bitflags,
                               xsize, ysize, index_list, &num_index);
                }
            }
        }
    }

    free(index_list);

    /* unwrap branch cut pixels */
    for (j=1; j<ysize; j++)
    {
        for (i=1; i<xsize; i++)
        {
            k = j*xsize + i;

            if (bitflags[k] & AVOID)
            {
                if (!(bitflags[k-1] & AVOID))
                {
                    soln[k] = soln[k-1] + Gradient(phase[k], phase[k-1]);
                    path_order[k] = ++n;
                }
                else if (!(bitflags[k-xsize] & AVOID))
                {
                    soln[k] = soln[k-xsize] + Gradient(phase[k], phase[k-xsize]);
                    path_order[k] = ++n;
                }

            }
        }
    }



    return num_pieces;
}




/* Unwrap the phase data (by Itoh's method) without crossing
 * any branch cuts.  Return number of disconnected pieces.
 */
int UnwrapAroundCutsFrontier(float *phase,
                             unsigned char *bitflags,
                             float *soln,
                             int xsize,
                             int ysize,
                             int *path_order,
                             float *grady,
                             float *gradx,
                             int *list,
                             int length,
                             int omp_avoid_pass)
{
    int    i, j, k, kk, x, y, l, index, n=0, num_pieces=0;
    int    flag, base_in, base_out, top_in, top_out;
    float  value;


    /* find starting point */

    for (k=0; k<length; k++)
    {
        if (!(*(bitflags + k) & (BRANCH_CUT | UNWRAPPED | BORDER)))
        {
        	++num_pieces;

        	/* starting pixel */
            soln[k] = phase[k];
            flag = 1;

            /* base and top indexes */
            base_in = 0;
            base_out = xsize + ysize;
            top_in = base_in;
            top_out = base_out;

            *(list + top_in++) = k;

            /* Solve each level */
            while (flag)
            {
            	/* start integration level */
                for (l=base_in; l<top_in; l++)
                {
                    kk = *(list + l);
                    x = kk%xsize;
                    y = kk/xsize;

                    *(bitflags + kk) |= UNWRAPPED;
                    value = *(soln + kk);

                    /* neighbor pixels */

                    index = kk - 1;

                    if (x-1 >= 0
                        && !(*(bitflags + index) & (BRANCH_CUT | UNWRAPPED | BORDER)))
                    {
                        /* solution */
                        *(bitflags + index) |= UNWRAPPED;
                        *(soln + index) = value + *(gradx + index);
                        *(list + top_out++) = index;
                    }


                    index = kk + 1;

                    if (x+1 < xsize
                        && !(*(bitflags + index) & (BRANCH_CUT | UNWRAPPED | BORDER)))
                    {
                        /* solution */
                    	*(bitflags + index) |= UNWRAPPED;
                        *(soln + index) = value - *(gradx + kk);
                        *(list + top_out++) = index;
                    }


                    index = kk - xsize;

                    if (y-1 >= 0
                        && !(*(bitflags + index) & (BRANCH_CUT | UNWRAPPED | BORDER)))
                    {
                        /* solution */
                    	*(bitflags + index) |= UNWRAPPED;
                        *(soln + index) = value + *(grady + index);
                        *(list + top_out++) = index;
                    }


                    index = kk + xsize;

                    if (y+1 < ysize
                        && !(*(bitflags + index) & (BRANCH_CUT | UNWRAPPED | BORDER)))
                    {
                        /* solution */
                    	*(bitflags + index) |= UNWRAPPED;
                        *(soln + index) = value - *(grady + kk);
                        *(list + top_out++) = index;
                    }

                }
                /* end of level loop for */


                /* Exchange limits of the list */
                if (base_out==top_out)
                    flag = 0;
                else
                {
                    swap(&base_in, &base_out);
                    swap(&top_in, &top_out);
                    top_out = base_out;
                }
            }
            /* end of in loop while */
        }
        /* end of starting pixel search if */

    }
    /* end of starting for */



    /* unwrap branch cut pixels (AVOID band: branch cuts + border) */
    if (omp_avoid_pass) {
        #pragma omp parallel for default(none) \
        private(i, j, k) \
        shared(ysize, xsize, bitflags, soln, phase)
        for (j = 1; j < ysize; j++) {
            for (i = 1; i < xsize; i++) {
                k = j * xsize + i;

                if (bitflags[k] & AVOID) {
                    if (!(bitflags[k - 1] & AVOID))
                        *(soln + k) = *(soln + k - 1) + Gradient(phase[k], phase[k - 1]);
                    else if (!(bitflags[k - xsize] & AVOID))
                        *(soln + k) = *(soln + k - xsize)
                                     + Gradient(phase[k], phase[k - xsize]);
                }
            }
        }
    } else {
        for (j = 1; j < ysize; j++) {
            for (i = 1; i < xsize; i++) {
                k = j * xsize + i;

                if (bitflags[k] & AVOID) {
                    if (!(bitflags[k - 1] & AVOID))
                        *(soln + k) = *(soln + k - 1) + Gradient(phase[k], phase[k - 1]);
                    else if (!(bitflags[k - xsize] & AVOID))
                        *(soln + k) = *(soln + k - xsize)
                                     + Gradient(phase[k], phase[k - xsize]);
                }
            }
        }
    }

    return num_pieces;
}




/* Place a branch cut in the bitflags array from pixel (a,b) */
/* to pixel (c,d).  The bit for the branch cut pixels is     */
/* given by the value of "code".                             */
void PlaceCut(unsigned char *array,
              int a,
              int b,
              int c,
              int d,
              int xsize,
              int ysize,
              int code)
{
    int  i, j, k, ii, jj, m, n, istep, jstep;
    double  r;

    /* residue location is upper-left corner of 4-square */
    if (c > a && a > 0) a++;
    else if (c < a && c > 0) c++;
    if (d > b && b > 0) b++;
    else if (d < b && d > 0) d++;

    if (a==c && b==d) {
        array[b*xsize + a] |= code;
        return;
    }
    m = (a < c) ? c - a : a - c;
    n = (b < d) ? d - b : b - d;
    if (m > n) {
        istep = (a < c) ? +1 : -1;
        r = ((double)(d - b))/((double)(c - a));
        for (i=a; i!=c+istep; i+=istep) {
            j = b + (i - a)*r + 0.5;
            array[j*xsize + i] |= code;
        }
    }
    else {   /* n < m */
        jstep = (b < d) ? +1 : -1;
        r = ((double)(c - a))/((double)(d - b));
        for (j=b; j!=d+jstep; j+=jstep) {
            i = a + (j - b)*r + 0.5;
            array[j*xsize + i] |= code;
        }
    }
    return;
}



/* Return the squared distance between the pixel (a,b) and the */
/* nearest border pixel.  The border pixels are encoded in the */
/* bitflags array by the value of "border_code".               */
int DistToBorder(unsigned char *bitflags,
                 int border_code,
                 int a,
                 int b,
                 int *ra,
                 int *rb, int xsize,
                 int ysize)
{
    int  besta, bestb, found, dist2, best_dist2;
    int  i, j, k, bs;
    *ra = *rb = 0;
    for (bs=0; bs<xsize + ysize; bs++) {
        found = 0;
        best_dist2 = 1000000;  /* initialize to large value */
        /* search boxes of increasing size until border pixel found */
        for (j=b - bs; j<=b + bs; j++) {
            for (i=a - bs; i<=a + bs; i++) {
                k = j*xsize + i;
                if (i<=0 || i>=xsize - 1 || j<=0 || j>=ysize - 1
                    || (bitflags[k] & border_code)) {
                    found = 1;
                    dist2 = (j - b)*(j - b) + (i - a)*(i - a);
                    if (dist2 < best_dist2) {
                        best_dist2 = dist2;
                        besta = i;
                        bestb = j;
                    }
                }
            }
        }
        if (found) {
            *ra = besta;
            *rb = bestb;
            break;
        }
    }
    return best_dist2;
}




/* Goldstein's phase-unwrapping algorithm.  The bitflags store */
/* the masked pixels (to be ignored) and the residues and      */
/* accumulates other info such as the branch cut pixels.       */
void BranchCuts_parallel(unsigned char *bitflags,
                         int MaxCutLen,
                         int NumRes,
                         int xsize,
                         int ysize,
                         int iniy,
                         int endy)
{
    int            i, j, k, ii, jj, kk, m, n, ri, rj;
    int            charge, boxctr_i, boxctr_j, boxsize, bs2;
    int            dist, min_dist, rim_i, rim_j, near_i, near_j;
    int            ka, num_active, max_active, *active_list;
    int            draw_cut_line;
    double         r;

    if (MaxCutLen < 2) MaxCutLen = 2;
    max_active = NumRes + 10;
    AllocateInt(&active_list, max_active + 1, "book keeping data");

    /* branch cuts */

    for (j=iniy; j<endy; j++)
    {
        for (i=0; i<xsize; i++)
        {
        	k = j*xsize + i;

            if ((bitflags[k] & (POS_RES | NEG_RES))
                && !(bitflags[k] & VISITED))
            {
                bitflags[k] |= VISITED;  /* turn on visited flag */
                bitflags[k] |= ACTIVE;   /* turn on active flag */
                charge = (bitflags[k] & POS_RES) ? 1 : -1;
                num_active = 0;
                active_list[num_active++] = k;

                if (num_active > max_active)
                    num_active = max_active;

                for (boxsize = 3; boxsize<=2*MaxCutLen; boxsize += 2)
                {
                    bs2 = boxsize/2;
                    for (ka=0; ka<num_active; ka++)
                    {
                        boxctr_i = active_list[ka]%xsize;
                        boxctr_j = active_list[ka]/xsize;
                        for (jj=boxctr_j - bs2; jj<=boxctr_j + bs2; jj++)
                        {
                            for (ii=boxctr_i - bs2; ii<=boxctr_i + bs2; ii++)
                            {
                                kk = jj*xsize + ii;
                                if (ii<0 || ii>=xsize || jj<0 || jj>=ysize)
                                {
                                    continue;
                                }
                                else
                                {
                                    if (ii==0 || ii==xsize-1 || jj==0 || jj==ysize-1
                                        || (bitflags[kk] & BORDER))
                                    {
                                        charge = 0;
                                        DistToBorder(bitflags, BORDER, boxctr_i,
                                                     boxctr_j, &ri, &rj, xsize, ysize);
                                        PlaceCut(bitflags, ri, rj, boxctr_i, boxctr_j,
                                                 xsize, ysize, BRANCH_CUT);
                                    }
                                    else if ((bitflags[kk] & (POS_RES | NEG_RES))
                                             && !(bitflags[kk] & ACTIVE))
                                    {
                                        if (!(bitflags[kk] & VISITED))
                                        {
                                            charge += (bitflags[kk] & POS_RES) ? 1 : -1;
                                            bitflags[kk] |= VISITED;   /* set flag */
                                        }
                                        active_list[num_active++] = kk;
                                        if (num_active > max_active)
                                            num_active = max_active;
                                        bitflags[kk] |= ACTIVE;  /* set active flag */
                                        PlaceCut(bitflags, ii, jj, boxctr_i, boxctr_j,
                                                 xsize, ysize, BRANCH_CUT);
                                    }
                                    if (charge==0)
                                        goto continue_scan;
                                }  /* else */
                            }   /* for (ii ... */
                        }   /* for (jj ... */
                    }  /* for (ka ... */
                }   /* for (boxsize ... */

                if (charge != 0)
                {   /* connect branch cuts to rim */
                    min_dist = xsize + ysize;  /* large value */
                    for (ka=0; ka<num_active; ka++)
                    {
                        ii = active_list[ka]%xsize;
                        jj = active_list[ka]/xsize;
                        if ((dist = DistToBorder(bitflags, BORDER,
                                                 ii, jj, &ri, &rj, xsize, ysize))<min_dist)
                        {
                            min_dist = dist;
                            near_i = ii;
                            near_j = jj;
                            rim_i = ri;
                            rim_j = rj;
                        }
                    }

                    PlaceCut(bitflags, near_i, near_j, rim_i, rim_j,
                             xsize, ysize, BRANCH_CUT);
                }
                continue_scan :
                /* mark all active pixels inactive */
                for (ka=0; ka<num_active; ka++)
                    bitflags[active_list[ka]] &= ~ACTIVE;  /* turn flag off */
            }  /* if (bitflags ... */

        }
    }



    free(active_list);

    return;
}




/* Goldstein's phase-unwrapping algorithm.  The bitflags store */
/* the masked pixels (to be ignored) and the residues and      */
/* accumulates other info such as the branch cut pixels.       */
void GoldsteinBranchCuts_parallel(unsigned char *bitflags,
                         int MaxCutLen,
                         int NumRes,
                         int xsize,
                         int ysize)
{
    int band, b, iniy, endy;
    int MaxCutLen2;


    /* length of a band */
    band = (int)ceil((double)ysize/(double)NUM_CORES);

    MaxCutLen2 = (xsize + band)/2;

    /*
     * Place branch cuts per band
     */
	#pragma omp parallel for default(none) \
	private(b, iniy, endy) \
	shared(NUM_CORES, band, bitflags, MaxCutLen2, NumRes, xsize, ysize)
    for (b=0; b<NUM_CORES; b++)
    {
    	iniy = b*band;

    	if (b<NUM_CORES-1)
    	    endy = iniy + band;
    	else
    	    endy = ysize;

    	BranchCuts_parallel(bitflags, MaxCutLen2, NumRes, xsize, ysize, iniy, endy);
    }
}




/* Goldstein's phase-unwrapping algorithm.  The bitflags store */
/* the masked pixels (to be ignored) and the residues and      */
/* accumulates other info such as the branch cut pixels.       */
void GoldsteinBranchCuts_serial(unsigned char *bitflags,
                         int MaxCutLen,
                         int NumRes,
                         int xsize,
                         int ysize)
{
    int            i, j, k, ii, jj, kk, m, n, ri, rj;
    int            charge, boxctr_i, boxctr_j, boxsize, bs2;
    int            dist, min_dist, rim_i, rim_j, near_i, near_j;
    int            ka, num_active, max_active, *active_list;
    double         r;

    if (MaxCutLen < 2) MaxCutLen = 2;
    max_active = NumRes + 10;
    AllocateInt(&active_list, max_active + 1, "book keeping data");

    /* branch cuts */

    for (j=0; j<ysize; j++)
    {
        for (i=0; i<xsize; i++)
        {
            k = j*xsize + i;
            if ((bitflags[k] & (POS_RES | NEG_RES))
                && !(bitflags[k] & VISITED))
            {
                bitflags[k] |= VISITED;  /* turn on visited flag */
                bitflags[k] |= ACTIVE;   /* turn on active flag */
                charge = (bitflags[k] & POS_RES) ? 1 : -1;
                num_active = 0;
                active_list[num_active++] = k;

                if (num_active > max_active)
                    num_active = max_active;

                for (boxsize = 3; boxsize<=2*MaxCutLen; boxsize += 2)
                {
                    bs2 = boxsize/2;
                    for (ka=0; ka<num_active; ka++)
                    {
                        boxctr_i = active_list[ka]%xsize;
                        boxctr_j = active_list[ka]/xsize;
                        for (jj=boxctr_j - bs2; jj<=boxctr_j + bs2; jj++)
                        {
                            for (ii=boxctr_i - bs2; ii<=boxctr_i + bs2; ii++)
                            {
                                kk = jj*xsize + ii;
                                if (ii<0 || ii>=xsize || jj<0 || jj>=ysize)
                                {
                                    continue;
                                }
                                else
                                {
                                    if (ii==0 || ii==xsize-1 || jj==0 || jj==ysize-1
                                        || (bitflags[kk] & BORDER))
                                    {
                                        charge = 0;
                                        DistToBorder(bitflags, BORDER, boxctr_i,
                                                     boxctr_j, &ri, &rj, xsize, ysize);
                                        PlaceCut(bitflags, ri, rj, boxctr_i, boxctr_j,
                                                 xsize, ysize, BRANCH_CUT);
                                    }
                                    else if ((bitflags[kk] & (POS_RES | NEG_RES))
                                             && !(bitflags[kk] & ACTIVE))
                                    {
                                        if (!(bitflags[kk] & VISITED))
                                        {
                                            charge += (bitflags[kk] & POS_RES) ? 1 : -1;
                                            bitflags[kk] |= VISITED;   /* set flag */
                                        }
                                        active_list[num_active++] = kk;
                                        if (num_active > max_active)
                                            num_active = max_active;
                                        bitflags[kk] |= ACTIVE;  /* set active flag */
                                        PlaceCut(bitflags, ii, jj, boxctr_i, boxctr_j,
                                                 xsize, ysize, BRANCH_CUT);
                                    }
                                    if (charge==0)
                                        goto continue_scan2;
                                }  /* else */
                            }   /* for (ii ... */
                        }   /* for (jj ... */
                    }  /* for (ka ... */
                }   /* for (boxsize ... */

                if (charge != 0)
                {   /* connect branch cuts to rim */
                    min_dist = xsize + ysize;  /* large value */
                    for (ka=0; ka<num_active; ka++)
                    {
                        ii = active_list[ka]%xsize;
                        jj = active_list[ka]/xsize;
                        if ((dist = DistToBorder(bitflags, BORDER,
                                                 ii, jj, &ri, &rj, xsize, ysize))<min_dist)
                        {
                            min_dist = dist;
                            near_i = ii;
                            near_j = jj;
                            rim_i = ri;
                            rim_j = rj;
                        }
                    }

                    PlaceCut(bitflags, near_i, near_j, rim_i, rim_j,
                             xsize, ysize, BRANCH_CUT);
                }
                continue_scan2 :
                /* mark all active pixels inactive */
                for (ka=0; ka<num_active; ka++)
                    bitflags[active_list[ka]] &= ~ACTIVE;  /* turn flag off */
            }  /* if (bitflags ... */
        }  /* for (i ... */
    }  /* for (j ... */


    free(active_list);

    return;
}



/* Detect residues in phase data and mark them as positive or  */
/* negative residues in the bitflags array.  Ignore the pixels */
/* marked with avoid_code in the bitflags araay.               */

int Residues_parallel(float *phase,
             unsigned char *bitflags,
             int xsize,
             int ysize)
{
    int  i, j, k, NumRes=0;
    double  r;

    #pragma omp parallel for default(none) \
    private(j, i, k, r) \
    shared(ysize, xsize, bitflags, phase) \
    reduction(+ : NumRes) \
    collapse(2)
    for (j=0; j<ysize - 1; j++)
    {
        for (i=0; i<xsize - 1; i++)
        {
            k = j*xsize + i;

            if (bitflags && ((bitflags[k] & AVOID)
                             || (bitflags[k+1] & AVOID)
                             || (bitflags[k+1+xsize] & AVOID)
                             || (bitflags[k+xsize] & AVOID))) {
                continue; /* masked region: don't unwrap */
            }
            r = Gradient(phase[k+1], phase[k])
            + Gradient(phase[k+1+xsize], phase[k+1])
            + Gradient(phase[k+xsize], phase[k+1+xsize])
            + Gradient(phase[k], phase[k+xsize]);
            if (bitflags) {
                if (r > RESIDUE_THRESHOLD) bitflags[k] |= POS_RES;
                else if (r < -RESIDUE_THRESHOLD) bitflags[k] |= NEG_RES;
            }
            if (r*r > RESIDUE_THRESHOLD * RESIDUE_THRESHOLD)
                ++NumRes;
        }
    }
    return NumRes;
}



/* Detect residues in phase data and mark them as positive or  */
/* negative residues in the bitflags array.  Ignore the pixels */
/* marked with avoid_code in the bitflags araay.               */

int Residues_serial(float *phase,
             unsigned char *bitflags,
             int xsize,
             int ysize)
{
    int  i, j, k, NumRes=0;
    double  r;

    for (j=0; j<ysize - 1; j++)
    {
        for (i=0; i<xsize - 1; i++)
        {
            k = j*xsize + i;

            if (bitflags && ((bitflags[k] & AVOID)
                             || (bitflags[k+1] & AVOID)
                             || (bitflags[k+1+xsize] & AVOID)
                             || (bitflags[k+xsize] & AVOID))) {
                continue; /* masked region: don't unwrap */
            }
            r = Gradient(phase[k+1], phase[k])
            + Gradient(phase[k+1+xsize], phase[k+1])
            + Gradient(phase[k+xsize], phase[k+1+xsize])
            + Gradient(phase[k], phase[k+xsize]);
            if (bitflags) {
                if (r > RESIDUE_THRESHOLD) bitflags[k] |= POS_RES;
                else if (r < -RESIDUE_THRESHOLD) bitflags[k] |= NEG_RES;
            }
            if (r*r > RESIDUE_THRESHOLD * RESIDUE_THRESHOLD)
                ++NumRes;
        }
    }
    return NumRes;
}


/* -----------------------------------------------------------------------
 *  Image I/O helpers  (float / uint8 TIFF via tiff_io.cpp: CImg + libtiff)
 * -------------------------------------------------------------------- */

/* Return 1 if path has a .tif / .tiff extension (case-insensitive). */
static int is_tiff_path(const char *path)
{
    const char *ext = strrchr(path, '.');
    if (!ext) return 0;
    return (strcasecmp(ext, ".tif")  == 0 ||
            strcasecmp(ext, ".tiff") == 0);
}

/*
 * Load float32 wrapped phase from TIFF (radians, principal ~[-PI, PI]).
 */
static float *load_phase_from_tiff(const char *path, int *xsize, int *ysize)
{
    int length;
    float *phase;

    float *img = tiff_io_load_float(path, xsize, ysize);
    if (!img) {
        fprintf(stderr, "Error: cannot load TIFF '%s'\n", path);
        exit(FILE_OPEN_ERROR);
    }

    length = (*xsize) * (*ysize);
    AllocateFloat(&phase, length, "phase from TIFF");
    memcpy(phase, img, (size_t)length * sizeof(float));
    free(img);

    printf("Loaded TIFF '%s' (%d x %d, float32 rad)\n",
           path, *xsize, *ysize);
    return phase;
}

/* Save unwrapped phase as float32 TIFF (radians, same dynamic range as soln). */
static void save_float_as_tiff(const char *path, const float *data,
                               int xsize, int ysize)
{
    if (tiff_io_save_float(path, data, xsize, ysize) != 0)
        fprintf(stderr, "Warning: failed to write float TIFF '%s'\n", path);
    else
        printf("Saved '%s'\n", path);
}

/*
 * Save a byte visualization as single-channel 8-bit TIFF.
 * Pixels whose bits overlap mask_code are written as 255, others as 0.
 */
static void save_byte_as_tiff(const char *path, const unsigned char *data,
                               int xsize, int ysize, int mask_code)
{
    int k, length = xsize * ysize;
    unsigned char mask = mask_code ? (unsigned char)mask_code : 0xFF;
    unsigned char *out = (unsigned char *)malloc((size_t)length);

    for (k = 0; k < length; k++)
        out[k] = (data[k] & mask) ? 255 : 0;

    if (tiff_io_save_u8(path, out, xsize, ysize) != 0)
        fprintf(stderr, "Warning: failed to write uint8 TIFF '%s'\n", path);
    else
        printf("Saved '%s'\n", path);
    free(out);
}


/* -----------------------------------------------------------------------
 *  Ground-truth comparison helpers
 * -------------------------------------------------------------------- */

static int parse_json_range(const char *json_path, double *lo, double *hi)
{
    FILE *fp = fopen(json_path, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open JSON '%s'\n", json_path);
        return -1;
    }
    char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    buf[n] = '\0';

    int found_lo = 0, found_hi = 0;
    char *p;

    p = strstr(buf, "\"true_lo\"");
    if (p) {
        p = strchr(p + 9, ':');
        if (p) { *lo = strtod(p + 1, NULL); found_lo = 1; }
    }
    p = strstr(buf, "\"true_hi\"");
    if (p) {
        p = strchr(p + 9, ':');
        if (p) { *hi = strtod(p + 1, NULL); found_hi = 1; }
    }

    if (!found_lo || !found_hi) {
        fprintf(stderr, "Error: could not find true_lo/true_hi in '%s'\n",
                json_path);
        return -1;
    }
    return 0;
}

/* Min/max of float array (for logging when JSON sidecar is absent). */
static void float_range_stats(const float *a, int n, double *lo, double *hi)
{
    int k;
    double mn = (double)a[0], mx = (double)a[0];
    for (k = 1; k < n; k++) {
        double v = (double)a[k];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    *lo = mn;
    *hi = mx;
}

/*
 * Ground truth: single-channel float32 TIFF, values already in radians
 * (same convention as Python generate_phase.save_case).
 */
static float *load_ground_truth_tiff(const char *path,
                                     int expected_w, int expected_h)
{
    int w, h;
    float *truth = tiff_io_load_float(path, &w, &h);

    if (!truth) {
        fprintf(stderr, "Error: cannot load ground-truth TIFF '%s'\n", path);
        return NULL;
    }
    if (w != expected_w || h != expected_h) {
        fprintf(stderr,
                "Error: ground truth size %dx%d != input size %dx%d\n",
                w, h, expected_w, expected_h);
        free(truth);
        return NULL;
    }

    printf("Loaded ground truth '%s' (%d x %d), float32 radians (TIFF)\n",
           path, w, h);
    return truth;
}

static double compute_rms(const float *result, const float *truth, int length)
{
    int k;
    double sum_diff = 0.0, offset, sum_sq = 0.0, d;

    for (k = 0; k < length; k++)
        sum_diff += (double)(result[k] - truth[k]);
    offset = sum_diff / length;

    for (k = 0; k < length; k++) {
        d = (double)(result[k] - truth[k]) - offset;
        sum_sq += d * d;
    }
    return sqrt(sum_sq / length);
}

static double wrap_to_pi_double(double x)
{
    while (x > M_PI)
        x -= 2.0 * M_PI;
    while (x < -M_PI)
        x += 2.0 * M_PI;
    return x;
}

/* Relaxed CPU-vs-GPU phase comparison.
 * Useful when CPU and GPU use different valid branch-cut topologies.
 * global_offset_rmse: allows one global constant offset.
 * wrapped_diff_rmse: additionally treats local 2*pi offsets as equivalent.
 * wrapped_grad_rmse: compares local wrapped gradients, which is usually the
 * strongest topology-agnostic metric for unwrapped phase quality.
 */
static void soln_relaxed_rms_stats(const float *gpu,
                                   const float *cpu,
                                   int xsize,
                                   int ysize,
                                   double *global_offset_rmse,
                                   double *wrapped_diff_rmse,
                                   double *wrapped_grad_rmse,
                                   int *wrapped_bad_count,
                                   double wrapped_tol)
{
    int k, x, y;
    int length = xsize * ysize;
    double offset = 0.0;
    double ss_global = 0.0, ss_wrap = 0.0, ss_grad = 0.0;
    int grad_n = 0, bad = 0;

    for (k = 0; k < length; k++)
        offset += (double)gpu[k] - (double)cpu[k];
    offset /= (double)length;

    for (k = 0; k < length; k++) {
        double d = ((double)gpu[k] - (double)cpu[k]) - offset;
        double dw = wrap_to_pi_double(d);
        ss_global += d * d;
        ss_wrap += dw * dw;
        if (fabs(dw) > wrapped_tol)
            ++bad;
    }

    for (y = 0; y < ysize; y++) {
        for (x = 0; x < xsize - 1; x++) {
            k = y * xsize + x;
            double gg = Gradient(gpu[k + 1], gpu[k]);
            double cg = Gradient(cpu[k + 1], cpu[k]);
            double d = wrap_to_pi_double(gg - cg);
            ss_grad += d * d;
            ++grad_n;
        }
    }
    for (y = 0; y < ysize - 1; y++) {
        for (x = 0; x < xsize; x++) {
            k = y * xsize + x;
            double gg = Gradient(gpu[k + xsize], gpu[k]);
            double cg = Gradient(cpu[k + xsize], cpu[k]);
            double d = wrap_to_pi_double(gg - cg);
            ss_grad += d * d;
            ++grad_n;
        }
    }

    *global_offset_rmse = sqrt(ss_global / (double)length);
    *wrapped_diff_rmse = sqrt(ss_wrap / (double)length);
    *wrapped_grad_rmse = grad_n ? sqrt(ss_grad / (double)grad_n) : 0.0;
    *wrapped_bad_count = bad;
}


/* Bitwise compare for parallel-vs-serial verification (masked flags only). */
static int count_flag_mismatch(const unsigned char *a, const unsigned char *b,
                               int n, unsigned char mask)
{
    int k, bad = 0;
    for (k = 0; k < n; k++)
        if ((a[k] & mask) != (b[k] & mask))
            ++bad;
    return bad;
}


/* Count pixels containing a selected flag bit. */
static int count_flag_pixels(const unsigned char *bf, int n, unsigned char mask)
{
    int k, c = 0;
    if (!bf)
        return 0;
    for (k = 0; k < n; k++)
        if (bf[k] & mask)
            ++c;
    return c;
}

/* Count branch-cut pixels that touch the image border.
 * This catches a common broken Stage-2 behavior where most residues fall back
 * to rim connection instead of neutralizing into local clusters.
 */
static int count_border_touching_cuts(const unsigned char *bf, int xsize, int ysize)
{
    int x, y, c = 0;
    if (!bf || xsize <= 0 || ysize <= 0)
        return 0;

    for (x = 0; x < xsize; x++) {
        if (bf[x] & BRANCH_CUT)
            ++c;
        if (ysize > 1 && (bf[(ysize - 1) * xsize + x] & BRANCH_CUT))
            ++c;
    }
    for (y = 1; y < ysize - 1; y++) {
        if (bf[y * xsize] & BRANCH_CUT)
            ++c;
        if (xsize > 1 && (bf[y * xsize + xsize - 1] & BRANCH_CUT))
            ++c;
    }
    return c;
}


/* Compare two unwrapped solutions (radians). */
static void soln_diff_stats(const float *a, const float *b, int n,
                            double *max_abs, int *n_gt_tol, float tol)
{
    int k;
    double mx = 0.0;
    int cnt = 0;

    for (k = 0; k < n; k++) {
        double d = fabs((double)a[k] - (double)b[k]);
        if (d > mx)
            mx = d;
        if ((float)d > tol)
            ++cnt;
    }
    *max_abs = mx;
    *n_gt_tol = cnt;
}


/* -----------------------------------------------------------------------
 *  Pluggable unwrap kernels (timed as one unit inside goldstein_phase_unwrapping)
 *
 *  UNWRAP_BACKEND_PARALLEL_CPU — OpenMP residues + parallel branch cuts +
 *                                frontier unwrap (omp_avoid_pass = 1).
 *  UNWRAP_BACKEND_SERIAL_CPU   — Serial residues + serial branch cuts +
 *                                frontier unwrap (omp_avoid_pass = 0).
 *  UNWRAP_BACKEND_CUDA_STUB    — All three stages (residue identification,
 *                                residue matching / branch cuts, and unwrap)
 *                                run on the GPU when device buffers are
 *                                available; falls back to CPU per-stage if not.
 * -------------------------------------------------------------------- */

#define UNWRAP_BACKEND_PARALLEL_CPU 0
#define UNWRAP_BACKEND_SERIAL_CPU   1
#define UNWRAP_BACKEND_CUDA_STUB    2
#define UNWRAP_BACKEND_COUNT        3

typedef struct UnwrapKernelCtx {
    float                 *phase;
    unsigned char         *bitflags;
    float                 *soln;
    int                    xsize;
    int                    ysize;
    int                    length;
    int                   *path_order;
    float                 *grady;
    float                 *gradx;
    int                   *list;
    UnwrapCudaDeviceBufs   cuda_dev;
} UnwrapKernelCtx;

typedef struct UnwrapKernelResult {
    int     num_residues;
    int     num_pieces;
    double  elapsed_ms;
    double  ms_residues;
    double  ms_branch_cuts;
    double  ms_unwrap;
    double  ms_cuda_residue_match;
} UnwrapKernelResult;

typedef void (*unwrap_kernel_run_fn)(UnwrapKernelCtx *ctx,
                                       int              verify_effective,
                                       unsigned char   *snap_after_res,
                                       unsigned char   *bf_preunwrap,
                                       UnwrapKernelResult *out);

static void run_unwrap_kernel_parallel_cpu(
    UnwrapKernelCtx *ctx,
    int verify_effective,
    unsigned char *snap_after_res,
    unsigned char *bf_preunwrap,
    UnwrapKernelResult *out)
{
    clock_t ta, tb;
    int MaxCutLen = (ctx->xsize + ctx->ysize) / 2;

    out->ms_cuda_residue_match = 0.0;

    ta = clock();
    out->num_residues = Residues_parallel(ctx->phase, ctx->bitflags,
                                          ctx->xsize, ctx->ysize);
    tb = clock();
    out->ms_residues = timediff(ta, tb);
    if (verify_effective && snap_after_res)
        memcpy(snap_after_res, ctx->bitflags, (size_t)ctx->length);

    ta = clock();
    GoldsteinBranchCuts_parallel(ctx->bitflags, MaxCutLen,
                                 out->num_residues, ctx->xsize, ctx->ysize);
    tb = clock();
    out->ms_branch_cuts = timediff(ta, tb);
    if (verify_effective && bf_preunwrap)
        memcpy(bf_preunwrap, ctx->bitflags, (size_t)ctx->length);

    ta = clock();
    out->num_pieces = UnwrapAroundCutsFrontier(
        ctx->phase, ctx->bitflags, ctx->soln,
        ctx->xsize, ctx->ysize, ctx->path_order,
        ctx->grady, ctx->gradx, ctx->list, ctx->length, 1);
    tb = clock();
    out->ms_unwrap = timediff(ta, tb);

    out->elapsed_ms = out->ms_residues + out->ms_branch_cuts + out->ms_unwrap;
}

static void run_unwrap_kernel_serial_cpu(
    UnwrapKernelCtx *ctx,
    int verify_effective,
    unsigned char *snap_after_res,
    unsigned char *bf_preunwrap,
    UnwrapKernelResult *out)
{
    clock_t ta, tb;
    int MaxCutLen = (ctx->xsize + ctx->ysize) / 2;

    out->ms_cuda_residue_match = 0.0;

    ta = clock();
    out->num_residues = Residues_serial(ctx->phase, ctx->bitflags,
                                        ctx->xsize, ctx->ysize);
    tb = clock();
    out->ms_residues = timediff(ta, tb);
    if (verify_effective && snap_after_res)
        memcpy(snap_after_res, ctx->bitflags, (size_t)ctx->length);

    ta = clock();
    GoldsteinBranchCuts_serial(ctx->bitflags, MaxCutLen,
                               out->num_residues, ctx->xsize, ctx->ysize);
    tb = clock();
    out->ms_branch_cuts = timediff(ta, tb);
    if (verify_effective && bf_preunwrap)
        memcpy(bf_preunwrap, ctx->bitflags, (size_t)ctx->length);

    ta = clock();
    out->num_pieces = UnwrapAroundCutsFrontier(
        ctx->phase, ctx->bitflags, ctx->soln,
        ctx->xsize, ctx->ysize, ctx->path_order,
        ctx->grady, ctx->gradx, ctx->list, ctx->length, 0);
    tb = clock();
    out->ms_unwrap = timediff(ta, tb);

    out->elapsed_ms = out->ms_residues + out->ms_branch_cuts + out->ms_unwrap;
}

static void run_unwrap_kernel_cuda(
    UnwrapKernelCtx *ctx,
    int verify_effective,
    unsigned char *snap_after_res,
    unsigned char *bf_preunwrap,
    UnwrapKernelResult *out)
{
    clock_t ta, tb;
    int     MaxCutLen   = (ctx->xsize + ctx->ysize) / 2;
    int     have_device = (ctx->cuda_dev.d_phase != NULL
                           && ctx->cuda_dev.d_bitflags != NULL
                           && ctx->cuda_dev.d_soln != NULL);

    out->ms_cuda_residue_match = 0.0;

    /* Stage 1: residue identification on the GPU. */
    if (have_device) {
        ta = clock();
        int rc = unwrap_cuda_launch_residue_identification(
            ctx->phase, ctx->bitflags, &ctx->cuda_dev,
            ctx->xsize, ctx->ysize, ctx->length);
        tb = clock();
        if (rc < 0) {
            fprintf(stderr,
                    "unwrap backend 'cuda': residue-identification kernel failed "
                    "(rc=%d); falling back to CPU.\n", rc);
            ta = clock();
            out->num_residues = Residues_serial(ctx->phase, ctx->bitflags,
                                                ctx->xsize, ctx->ysize);
            tb = clock();
        } else {
            out->num_residues = rc;
        }
        out->ms_residues = timediff(ta, tb);
    } else {
        ta = clock();
        out->num_residues = Residues_serial(ctx->phase, ctx->bitflags,
                                            ctx->xsize, ctx->ysize);
        tb = clock();
        out->ms_residues = timediff(ta, tb);
    }

    if (verify_effective && snap_after_res)
        memcpy(snap_after_res, ctx->bitflags, (size_t)ctx->length);

    /* Stage 2: residue matching / branch cuts on the GPU. */
    if (have_device) {
        ta = clock();
        unwrap_cuda_launch_residue_matching(ctx->bitflags, &ctx->cuda_dev, MaxCutLen,
                                            out->num_residues, ctx->xsize, ctx->ysize,
                                            ctx->length);
        // GoldsteinBranchCuts_serial(ctx->bitflags, MaxCutLen, out->num_residues,
        //                            ctx->xsize, ctx->ysize);
        tb = clock();
        out->ms_branch_cuts = timediff(ta, tb);
    } else {
        fprintf(stderr,
                "unwrap backend 'cuda': no device buffers; running CPU "
                "branch-cut fallback.\n");
        ta = clock();
        GoldsteinBranchCuts_serial(ctx->bitflags, MaxCutLen, out->num_residues,
                                   ctx->xsize, ctx->ysize);
        tb = clock();
        out->ms_branch_cuts = timediff(ta, tb);
    }

    if (verify_effective && bf_preunwrap)
        memcpy(bf_preunwrap, ctx->bitflags, (size_t)ctx->length);

    /* Stage 3: unwrap on the GPU (tile-frontier or block-wise per build flag). */
    if (have_device) {
        ta = clock();
        unwrap_cuda_launch_unwrapping(
            ctx->phase, ctx->bitflags, ctx->soln,
            ctx->gradx, ctx->grady,
            &ctx->cuda_dev,
            ctx->xsize, ctx->ysize, ctx->length);
        tb = clock();
        out->ms_unwrap = timediff(ta, tb);
        /* The CUDA unwrap path does not enumerate connected components the way
         * UnwrapAroundCutsFrontier does; report 0 so the field stays defined. */
        out->num_pieces = 0;
    } else {
        ta = clock();
        out->num_pieces = UnwrapAroundCutsFrontier(
            ctx->phase, ctx->bitflags, ctx->soln, ctx->xsize, ctx->ysize,
            ctx->path_order, ctx->grady, ctx->gradx, ctx->list, ctx->length, 0);
        tb = clock();
        out->ms_unwrap = timediff(ta, tb);
    }

    out->elapsed_ms = out->ms_residues + out->ms_cuda_residue_match
        + out->ms_branch_cuts + out->ms_unwrap;
}

static const unwrap_kernel_run_fn g_unwrap_kernel_runners[UNWRAP_BACKEND_COUNT] = {
    run_unwrap_kernel_parallel_cpu,
    run_unwrap_kernel_serial_cpu,
    run_unwrap_kernel_cuda,
};

static const char *g_unwrap_backend_names[UNWRAP_BACKEND_COUNT] = {
    "parallel_cpu",
    "serial_cpu",
    "cuda",
};

/* -----------------------------------------------------------------------
 *  Core phase-unwrapping pipeline
 *
 *  input_path    – full path to the input file.
 *                  Supported: .tif / .tiff  (float32 wrapped phase, radians;
 *                             principal ~[-PI, PI]; xsize/ysize auto-detected)
 *                             any other extension   (raw binary read via
 *                             GetPhase; xsize and ysize must be provided)
 *
 *  output_prefix – path prefix used for output files (no extension).
 *                  Always writes *_unwrapped.tif (float32 radians).
 *                  With verify_serial: *_residues.tif and *_branchcuts.tif (debug).
 *
 *  type          – binary-format selector passed to GetPhase (ignored for
 *                  image inputs):
 *                    0 = 8-byte complex,  1 = 4-byte complex,
 *                    2 = 1-byte quantised phase,  3 = 4-byte float phase
 *
 *  xsize, ysize  – dimensions for binary inputs; pass 0 for TIFF inputs
 *                  (values are filled in by load_phase_from_tiff).
 *
 *  mask_flag     – 1 = load a mask from <output_prefix>.mask
 *
 *  gt_path       – path to float32 ground-truth TIFF in radians (or NULL).
 *  gt_lo, gt_hi  – optional metadata from JSON (for logging only; RMS uses
 *                  samples read directly from the TIFF).
 *  gt_json_valid – 1 if true_lo/true_hi were read from JSON successfully.
 *  verify_serial  – if non-zero, run serial-reference checks for the selected
 *                   backend: residue flags vs Residues_serial; branch layout
 *                   vs GoldsteinBranchCuts_serial and vs parallel@1 thread from
 *                   the same post-residue snapshot; unwrap vs a serial AVOID-band
 *                   replay from the same pre-unwrap bitflags.
 *
 *  unwrap_backend – UNWRAP_BACKEND_PARALLEL_CPU (default OpenMP path),
 *                   UNWRAP_BACKEND_SERIAL_CPU (serial reference), or
 *                   UNWRAP_BACKEND_CUDA_STUB (CPU residues / cuts / unwrap; CUDA
 *                   residue-matching kernel only when device buffers exist).
 *
 *  Return value  – elapsed ms for the selected unwrap kernel (sum of residue,
 *                  optional CUDA match, branch cuts, and frontier unwrap).
 *                  A timing report also prints host I/O, prep, verify, and output.
 * -------------------------------------------------------------------- */
double goldstein_phase_unwrapping(const char *input_path,
                                   const char *output_prefix,
                                   int type,
                                   int xsize,
                                   int ysize,
                                   int mask_flag,
                                   const char *gt_path,
                                   double gt_lo,
                                   double gt_hi,
                                   int gt_json_valid,
                                   int verify_serial,
                                   int unwrap_backend)
{
    int           *path_order;
    float         *phase;
    float         *soln;
    float         *grady, *gradx;
    float         *mask;
    unsigned char *unwrap, *bitflags;
    double         elapsed_time;
    double         ms_load_phase = 0.0, ms_load_mask = 0.0, ms_bitflags_init = 0.0;
    double         ms_gradxy = 0.0, ms_cuda_setup = 0.0, ms_verify = 0.0;
    double         ms_write_tiff = 0.0, ms_gt_rms = 0.0;
    double         ms_k_residues = 0.0, ms_k_branch = 0.0, ms_k_unwrap = 0.0;
    double         ms_k_cuda_match = 0.0;
    clock_t        _t0, _t1;
    FILE          *ifp, *ifm;
    char           fname[PATH_MAX];
    int            k, length, num_pieces, NumRes, MaxCutLen;
    int            *list;
    int            mis_res = 0, mis_brc = 0, mis_brc_cpu_1t = 0;
    int            n_soln_bad = 0, n_soln_bad_gold = 0, saved_nc = 0;
    int            branch_cuda = 0, branch_cpu = 0;
    int            border_cuda = 0, border_cpu = 0;
    double         soln_max_abs = 0.0, soln_max_abs_gold = 0.0;
    double         relaxed_global_rmse = 0.0, relaxed_wrapped_rmse = 0.0;
    double         relaxed_grad_rmse = 0.0;
    int            relaxed_wrapped_bad = 0;
    unsigned char *bf_res_ser = NULL, *bf_brc_ser = NULL, *bf_brc_1t = NULL;
    unsigned char *bf_preunwrap = NULL, *bf_unwrap2 = NULL, *bf_gold = NULL;
    unsigned char *snap_after_res = NULL;
    float          *soln_ser = NULL, *soln_gold = NULL;
    int            *path_order_ser = NULL, *path_order_gold = NULL;

    int is_tiff = is_tiff_path(input_path);
    int verify_effective;

    if (unwrap_backend < 0 || unwrap_backend >= UNWRAP_BACKEND_COUNT) {
        fprintf(stderr,
                "Warning: invalid unwrap backend id %d; using parallel_cpu.\n",
                unwrap_backend);
        unwrap_backend = UNWRAP_BACKEND_PARALLEL_CPU;
    }

    verify_effective = verify_serial;
    if (unwrap_backend == UNWRAP_BACKEND_CUDA_STUB)
        verify_effective = 0;  /* max-perf CUDA path keeps intermediate data on device */

    /* ---- For TIFF inputs load now to discover dimensions ---- */
    float *img_phase = NULL;
    if (is_tiff) {
        _t0 = clock();
        img_phase = load_phase_from_tiff(input_path, &xsize, &ysize);
        _t1 = clock();
        ms_load_phase = timediff(_t0, _t1);
    }

    /* ---- Allocate working arrays ---- */
    length = xsize * ysize;

    AllocateFloat(&phase,      length,          "phase data");
    AllocateFloat(&soln,       length,          "solution array");
    AllocateFloat(&grady,      length,          "vertical gradient");
    AllocateFloat(&gradx,      length,          "horizontal gradient");
    AllocateByte (&unwrap,     length,          "unwrap flag array");
    AllocateByte (&bitflags,   length,          "bitflag array");
    AllocateInt  (&path_order, length,          "integration path");
    AllocateFloat(&mask,       length,          "mask array");
    AllocateInt  (&list,       2*(xsize+ysize), "in-out list");

    /* ---- Read mask ---- */
    if (mask_flag) {
        _t0 = clock();
        snprintf(fname, sizeof(fname), "%s.mask", output_prefix);
        OpenFile(&ifm, fname, "rb");
        GetPhase(type, ifm, fname, mask, xsize, ysize);
        for (k = 0; k < length; k++)
            mask[k] = (mask[k] > 0) ? 1.0f : 0.0f;
        _t1 = clock();
        ms_load_mask = timediff(_t0, _t1);
    } else {
        for (k = 0; k < length; k++)
            mask[k] = 1.0f;
    }

    /* ---- Read phase ---- */
    if (is_tiff) {
        /* Transfer the pre-loaded data and release the temporary buffer */
        memcpy(phase, img_phase, length * sizeof(float));
        free(img_phase);
        img_phase = NULL;
    } else {
        /* Binary file: delegate to the existing reader */
        _t0 = clock();
        strncpy(fname, input_path, sizeof(fname) - 1);
        fname[sizeof(fname) - 1] = '\0';
        OpenFile(&ifp, fname, "rb");
        GetPhase(type, ifp, fname, phase, xsize, ysize);
        _t1 = clock();
        ms_load_phase = timediff(_t0, _t1);
    }

    /* ---- Initialise bitflags from mask ---- */
    _t0 = clock();
    #pragma omp parallel for default(none) \
    private(k) \
    shared(length, mask, bitflags)
    for (k = 0; k < length; k++)
        bitflags[k] = (mask[k] == 0.0f) ? BORDER : 0;
    _t1 = clock();
    ms_bitflags_init = timediff(_t0, _t1);

    /* ---- Pre-compute x/y gradients ----
       CUDA max-perf path computes gradx/grady in the fused Stage-1 kernel. */
    if (unwrap_backend != UNWRAP_BACKEND_CUDA_STUB) {
        _t0 = clock();
        Gradxy(phase, gradx, grady, xsize, ysize);
        _t1 = clock();
        ms_gradxy = timediff(_t0, _t1);
    } else {
        ms_gradxy = 0.0;
    }

    MaxCutLen = (xsize + ysize) / 2;

    if (verify_effective) {
        snap_after_res = (unsigned char *)malloc((size_t)length);
        if (!snap_after_res)
            fprintf(stderr,
                    "Warning: verify_serial: residue snapshot malloc failed.\n");
        bf_preunwrap = (unsigned char *)malloc((size_t)length);
        bf_unwrap2 = (unsigned char *)malloc((size_t)length);
        soln_ser = (float *)malloc((size_t)length * sizeof(float));
        path_order_ser = (int *)malloc((size_t)length * sizeof(int));
        if (!bf_preunwrap || !bf_unwrap2 || !soln_ser || !path_order_ser) {
            fprintf(stderr, "verify_serial: malloc failed (unwrap buffers)\n");
            free(bf_preunwrap);
            free(bf_unwrap2);
            free(soln_ser);
            free(path_order_ser);
            bf_preunwrap = NULL;
            bf_unwrap2 = NULL;
            soln_ser = NULL;
            path_order_ser = NULL;
        }
    }

    /* ---- Pluggable unwrap kernel (CUDA device malloc not timed below) ---- */
    {
        UnwrapKernelCtx     kctx;
        UnwrapKernelResult kres;

        memset(&kctx, 0, sizeof(kctx));
        kctx.phase       = phase;
        kctx.bitflags    = bitflags;
        kctx.soln        = soln;
        kctx.xsize       = xsize;
        kctx.ysize       = ysize;
        kctx.length      = length;
        kctx.path_order  = path_order;
        kctx.grady       = grady;
        kctx.gradx       = gradx;
        kctx.list        = list;

        if (unwrap_backend == UNWRAP_BACKEND_CUDA_STUB) {
            int cuda_alloc_rc;
            _t0 = clock();
            if (unwrap_cuda_init() != 0)
                fprintf(stderr,
                        "unwrap backend 'cuda': init failed (device buffers "
                        "not allocated).\n");
            else if ((cuda_alloc_rc = unwrap_cuda_device_bufs_alloc(length, &kctx.cuda_dev))
                     != 0)
                fprintf(stderr,
                        "unwrap backend 'cuda': cudaMalloc failed (cuda error %d)\n",
                        cuda_alloc_rc);
            _t1 = clock();
            ms_cuda_setup = timediff(_t0, _t1);
        }

        /* ---- Timed: pluggable unwrap kernel ---- */
        memset(&kres, 0, sizeof(kres));
        g_unwrap_kernel_runners[unwrap_backend](
            &kctx, verify_effective, snap_after_res, bf_preunwrap, &kres);
        NumRes       = kres.num_residues;
        num_pieces   = kres.num_pieces;
        elapsed_time = kres.elapsed_ms;
        ms_k_residues   = kres.ms_residues;
        ms_k_branch     = kres.ms_branch_cuts;
        ms_k_unwrap     = kres.ms_unwrap;
        ms_k_cuda_match = kres.ms_cuda_residue_match;

        if (unwrap_backend == UNWRAP_BACKEND_CUDA_STUB)
            unwrap_cuda_device_bufs_free(&kctx.cuda_dev);
    }

    if (unwrap_backend == UNWRAP_BACKEND_CUDA_STUB)
        printf("Number of residues: device-resident (host count intentionally not copied)\n");
    else
        printf("Number of residues: %d\n", NumRes);

    if (verify_effective) {
        _t0 = clock();

        /* 1) Verify Stage 1 residue flags against CPU serial residue pass. */
        bf_res_ser = (unsigned char *)malloc((size_t)length);
        if (bf_res_ser && snap_after_res) {
            for (k = 0; k < length; k++)
                bf_res_ser[k] = (mask[k] == 0.0f) ? BORDER : 0;
            Residues_serial(phase, bf_res_ser, xsize, ysize);
            mis_res = count_flag_mismatch(snap_after_res, bf_res_ser, length,
                                          (unsigned char)(POS_RES | NEG_RES));
        }
        free(bf_res_ser);
        bf_res_ser = NULL;

        /* 2) Build CPU serial branch-cut reference from post-residue snapshot.
         * Compare backend/CUDA Stage 2 output (bf_preunwrap) directly against it.
         */
        bf_brc_ser = (unsigned char *)malloc((size_t)length);
        bf_brc_1t = (unsigned char *)malloc((size_t)length);
        if (bf_brc_ser && snap_after_res) {
            memcpy(bf_brc_ser, snap_after_res, (size_t)length);
            GoldsteinBranchCuts_serial(bf_brc_ser, MaxCutLen, NumRes, xsize, ysize);

            branch_cpu = count_flag_pixels(bf_brc_ser, length, BRANCH_CUT);
            border_cpu = count_border_touching_cuts(bf_brc_ser, xsize, ysize);

            if (bf_preunwrap) {
                mis_brc = count_flag_mismatch(
                    bf_preunwrap, bf_brc_ser, length,
                    (unsigned char)(BRANCH_CUT | BORDER | POS_RES | NEG_RES));
                branch_cuda = count_flag_pixels(bf_preunwrap, length, BRANCH_CUT);
                border_cuda = count_border_touching_cuts(bf_preunwrap, xsize, ysize);
            }
        }

        /* Optional CPU-reference sanity: parallel@1 should match CPU serial.
         * This is not a CUDA Stage-2 check; it only validates CPU reference setup.
         */
        if (bf_brc_ser && bf_brc_1t && snap_after_res) {
            memcpy(bf_brc_1t, snap_after_res, (size_t)length);
            saved_nc = NUM_CORES;
            NUM_CORES = 1;
            omp_set_num_threads(1);
            GoldsteinBranchCuts_parallel(bf_brc_1t, MaxCutLen, NumRes, xsize, ysize);
            NUM_CORES = saved_nc;
            omp_set_num_threads(NUM_CORES);

            mis_brc_cpu_1t = count_flag_mismatch(
                bf_brc_1t, bf_brc_ser, length,
                (unsigned char)(BRANCH_CUT | BORDER | POS_RES | NEG_RES));
        }

        /* 3) Stage-3-only check: CPU unwrap replay on backend branch cuts. */
        if (bf_preunwrap && bf_unwrap2 && soln_ser && path_order_ser) {
            memset(soln_ser, 0, (size_t)length * sizeof(float));
            memset(path_order_ser, 0, (size_t)length * sizeof(int));
            memcpy(bf_unwrap2, bf_preunwrap, (size_t)length);
            UnwrapAroundCutsFrontier(phase, bf_unwrap2, soln_ser,
                                     xsize, ysize, path_order_ser,
                                     grady, gradx, list, length, 0);
            soln_diff_stats(soln, soln_ser, length,
                            &soln_max_abs, &n_soln_bad, 1e-5f);
        }

        /* 4) Full-pipeline check against CPU serial Stage2 + CPU unwrap.
         * This catches bad Stage-2 topology even if Stage 3 is correct under the
         * same wrong branch cuts.
         */
        if (bf_brc_ser) {
            bf_gold = (unsigned char *)malloc((size_t)length);
            soln_gold = (float *)malloc((size_t)length * sizeof(float));
            path_order_gold = (int *)malloc((size_t)length * sizeof(int));
            if (bf_gold && soln_gold && path_order_gold) {
                memset(soln_gold, 0, (size_t)length * sizeof(float));
                memset(path_order_gold, 0, (size_t)length * sizeof(int));
                memcpy(bf_gold, bf_brc_ser, (size_t)length);
                UnwrapAroundCutsFrontier(phase, bf_gold, soln_gold,
                                         xsize, ysize, path_order_gold,
                                         grady, gradx, list, length, 0);
                soln_diff_stats(soln, soln_gold, length,
                                &soln_max_abs_gold, &n_soln_bad_gold, 1e-5f);
                soln_relaxed_rms_stats(soln, soln_gold, xsize, ysize,
                                       &relaxed_global_rmse,
                                       &relaxed_wrapped_rmse,
                                       &relaxed_grad_rmse,
                                       &relaxed_wrapped_bad,
                                       0.10);
            } else {
                fprintf(stderr,
                        "verify_serial: malloc failed (gold reference buffers)\n");
            }
            free(bf_gold);
            free(soln_gold);
            free(path_order_gold);
            bf_gold = NULL;
            soln_gold = NULL;
            path_order_gold = NULL;
        }

        printf("\n=== Correctness (--verify-serial; backend=%s) ===\n",
               g_unwrap_backend_names[unwrap_backend]);
        printf("  Residues (POS|NEG)                  : %s  (%d mismatched cells)\n",
               mis_res ? "CHECK" : "PASS", mis_res);
        printf("  Branch layout backend vs CPU serial : %s  (%d mismatched cells)\n",
               mis_brc ? "CHECK" : "PASS", mis_brc);
        printf("  Branch layout CPU parallel@1 vs serial: %s  (%d mismatched cells)\n",
               mis_brc_cpu_1t ? "CHECK" : "PASS", mis_brc_cpu_1t);
        printf("  Branch pixels backend / CPU serial  : %d / %d\n",
               branch_cuda, branch_cpu);
        printf("  Border-touching cuts backend / CPU  : %d / %d\n",
               border_cuda, border_cpu);
        printf("  Stage3 only, kernel vs CPU replay on backend cuts : %s  "
               "(max |Delta| = %.6g, cells > 1e-5: %d)\n",
               n_soln_bad ? "CHECK" : "PASS", soln_max_abs, n_soln_bad);
        printf("  Full pipeline vs CPU serial Stage2+unwrap : %s  "
               "(max |Delta| = %.6g, cells > 1e-5: %d)\n",
               n_soln_bad_gold ? "CHECK" : "PASS",
               soln_max_abs_gold, n_soln_bad_gold);
        printf("  Relaxed RMS vs CPU serial output         : global=%.6g rad, wrapped=%.6g rad, wrapped-grad=%.6g rad, |wrapped Delta|>0.1: %d\n",
               relaxed_global_rmse, relaxed_wrapped_rmse,
               relaxed_grad_rmse, relaxed_wrapped_bad);

        free(bf_brc_ser);
        bf_brc_ser = NULL;
        free(bf_brc_1t);
        bf_brc_1t = NULL;

        _t1 = clock();
        ms_verify = timediff(_t0, _t1);
    }

    _t0 = clock();
    if (verify_effective && snap_after_res) {
        snprintf(fname, sizeof(fname), "%s_residues.tif", output_prefix);
        save_byte_as_tiff(fname, snap_after_res, xsize, ysize, RESIDUE);
    }
    free(snap_after_res);

    if (verify_effective) {
        snprintf(fname, sizeof(fname), "%s_branchcuts.tif", output_prefix);
        save_byte_as_tiff(fname, bitflags, xsize, ysize, BRANCH_CUT | BORDER);
    }

    free(bf_preunwrap);
    free(bf_unwrap2);
    free(soln_ser);
    free(path_order_ser);

    printf("Number of pieces: %d\n", num_pieces);

    /* ---- Save unwrapped phase (float32 radians) ---- */
    snprintf(fname, sizeof(fname), "%s_unwrapped.tif", output_prefix);
    save_float_as_tiff(fname, soln, xsize, ysize);
    _t1 = clock();
    ms_write_tiff = timediff(_t0, _t1);

    /* ---- RMS test against ground truth (float32 radians TIFF) ---- */
    if (gt_path) {
        _t0 = clock();
        float *truth = load_ground_truth_tiff(gt_path, xsize, ysize);
        if (truth) {
            double rms = compute_rms(soln, truth, length);
            double tr_lo, tr_hi;
            float_range_stats(truth, length, &tr_lo, &tr_hi);

            printf("=== RMS Test ===\n");
            printf("  Ground truth : %s\n", gt_path);
            printf("  File range   : [%.6f, %.6f] rad (float32 TIFF)\n",
                   tr_lo, tr_hi);
            if (gt_json_valid)
                printf("  JSON range   : [%.6f, %.6f] rad (metadata)\n",
                       gt_lo, gt_hi);
            printf("  RMS error    : %.6f rad  (%.4f deg)\n",
                   rms, rms * 180.0 / M_PI);
            printf("  Method       : mean-offset RMSE vs truth "
                   "(same as prior uint8-PNG pipeline, without 8-bit decode)\n");
            free(truth);
        }
        _t1 = clock();
        ms_gt_rms = timediff(_t0, _t1);
    }

    printf("\n=== Timing report (ms, clock()) ===\n");
    printf("  %-28s %12.3f\n", "Load wrapped phase", ms_load_phase);
    if (mask_flag)
        printf("  %-28s %12.3f\n", "Load mask", ms_load_mask);
    printf("  %-28s %12.3f\n", "Init bitflags (from mask)", ms_bitflags_init);
    printf("  %-28s %12.3f\n", "Gradxy", ms_gradxy);
    if (unwrap_backend == UNWRAP_BACKEND_CUDA_STUB)
        printf("  %-28s %12.3f\n", "CUDA init + device malloc", ms_cuda_setup);
    printf("  --- unwrap kernel (%s) ---\n",
           g_unwrap_backend_names[unwrap_backend]);
    printf("  %-28s %12.3f\n", "  Residues", ms_k_residues);
    printf("  %-28s %12.3f\n", "  Branch cuts", ms_k_branch);
    printf("  %-28s %12.3f\n", "  Unwrap (frontier)", ms_k_unwrap);
    printf("  %-28s %12.3f\n", "  Kernel subtotal", elapsed_time);
    if (verify_effective)
        printf("  %-28s %12.3f\n", "verify_serial (CPU checks)", ms_verify);
    printf("  %-28s %12.3f\n", "Write output TIFF(s)", ms_write_tiff);
    if (gt_path)
        printf("  %-28s %12.3f\n", "Ground truth load + RMS", ms_gt_rms);
    printf("\n");

    /* ---- Deallocate ---- */
    free(phase);
    free(soln);
    free(unwrap);
    free(path_order);
    free(grady);
    free(gradx);
    free(list);
    free(bitflags);
    free(mask);

    return elapsed_time;
}



static int parse_unwrap_backend(const char *s)
{
    if (!s)
        return -1;
    if (!strcasecmp(s, "parallel") || !strcasecmp(s, "parallel_cpu"))
        return UNWRAP_BACKEND_PARALLEL_CPU;
    if (!strcasecmp(s, "serial") || !strcasecmp(s, "serial_cpu"))
        return UNWRAP_BACKEND_SERIAL_CPU;
    if (!strcasecmp(s, "cuda") || !strcasecmp(s, "cuda_stub"))
        return UNWRAP_BACKEND_CUDA_STUB;
    return -1;
}


static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s -i <input.tif> [-g <truth.tif> [-j <meta.json>]] [-m] [-t <threads>] [-v] "
        "[-B <backend>]\n"
        "  %s                 (default built-in binary test)\n"
        "\n"
        "Options:\n"
        "  -i, --input   <path>   Wrapped phase: float32 TIFF, radians ~[-pi, pi]\n"
        "  -g, --ground  <path>   Ground truth: float32 radians TIFF (optional RMS)\n"
        "  -j, --json    <path>   Optional JSON with true_lo / true_hi (metadata)\n"
        "  -o, --output  <dir>    Output directory for <stem>_unwrapped.tif (mkdir -p).\n"
        "                         With -v, also <stem>_residues.tif and <stem>_branchcuts.tif.\n"
        "                         Default: input directory; stem from basename.\n"
        "  -m, --mask             Enable mask loading (<prefix>.mask)\n"
        "  -t, --threads <n>      Number of OpenMP threads\n"
        "  -v, --verify-serial    Serial cross-checks for any -B backend;\n"
        "                         extra work/memory; debug residue/branch TIFFs\n"
        "  -B, --backend <name>   Unwrap kernel: parallel_cpu (default), serial_cpu,\n"
        "                         or cuda_stub (CPU pipeline + CUDA matching hook)\n"
        "  -h, --help             Show this help message\n"
        "\n"
        "Examples:\n"
        "  %s -i phase_data/noisy_wrapped.tif\n"
        "  %s -i phase_data/noisy_wrapped.tif -g phase_data/noisy_true.tif "
        "-j phase_data/noisy.json\n",
        prog, prog, prog, prog);
}


int main(int argc, char *argv[])
{
    int    mask_flag = 0;
    int    type      = 3;
    int    num_threads = 0;
    int    verify_serial = 0;
    int    unwrap_backend = UNWRAP_BACKEND_PARALLEL_CPU;
    double elapsed_time = 0.0;
    int    i, opt;

    const char *input_path  = NULL;
    const char *gt_path     = NULL;
    const char *json_path   = NULL;
    const char *out_dir     = NULL;

    static struct option long_opts[] = {
        {"input",   required_argument, NULL, 'i'},
        {"ground",  required_argument, NULL, 'g'},
        {"json",    required_argument, NULL, 'j'},
        {"output",  required_argument, NULL, 'o'},
        {"mask",    no_argument,       NULL, 'm'},
        {"threads", required_argument, NULL, 't'},
        {"verify-serial", no_argument, NULL, 'v'},
        {"backend", required_argument, NULL, 'B'},
        {"help",    no_argument,       NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    while ((opt = getopt_long(argc, argv, "i:g:j:o:mt:vhB:", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'i': input_path  = optarg; break;
        case 'g': gt_path     = optarg; break;
        case 'j': json_path   = optarg; break;
        case 'o': out_dir     = optarg; break;
        case 'm': mask_flag   = 1;      break;
        case 't': num_threads = atoi(optarg); break;
        case 'v': verify_serial = 1;  break;
        case 'B':
            unwrap_backend = parse_unwrap_backend(optarg);
            if (unwrap_backend < 0) {
                fprintf(stderr, "Error: unknown --backend '%s'\n", optarg);
                print_usage(argv[0]);
                return BAD_USAGE;
            }
            break;
        case 'h': print_usage(argv[0]); return 0;
        default:  print_usage(argv[0]); return BAD_USAGE;
        }
    }

    /* Also accept a bare positional argument as the input image (legacy) */
    if (!input_path && optind < argc)
        input_path = argv[optind];

    /* ---- Thread setup ---- */
    if (num_threads <= 0) {
        NUM_CORES = omp_get_num_procs() / 2;
        if (NUM_CORES < 1) NUM_CORES = 1;
    } else {
        NUM_CORES = num_threads;
    }
    omp_set_num_threads(NUM_CORES);
    printf("Number of threads: %d\n", NUM_CORES);
    printf("Unwrap backend: %s\n", g_unwrap_backend_names[unwrap_backend]);

    if (input_path) {
        /* ---- TIFF image mode ---- */
        if (!is_tiff_path(input_path)) {
            fprintf(stderr,
                    "Error: '%s' is not a supported TIFF file "
                    "(.tif, .tiff).\n", input_path);
            print_usage(argv[0]);
            return BAD_USAGE;
        }

        /* Output prefix: default = full input path sans extension; or <out_dir>/<stem> */
        char default_prefix[PATH_MAX];
        char out_prefix_buf[PATH_MAX];
        char stem[PATH_MAX];
        const char *out_prefix;

        input_path_stem(input_path, stem, sizeof(stem));

        if (!out_dir) {
            strncpy(default_prefix, input_path, sizeof(default_prefix) - 1);
            default_prefix[sizeof(default_prefix) - 1] = '\0';
            {
                char *dot = strrchr(default_prefix, '.');
                if (dot)
                    *dot = '\0';
            }
            out_prefix = default_prefix;
        } else {
            if (mkdir_p(out_dir) != 0) {
                fprintf(stderr,
                        "Error: cannot create output directory '%s': %s\n",
                        out_dir, strerror(errno));
                return FILE_WRITE_ERROR;
            }
            if (!path_is_existing_dir(out_dir)) {
                fprintf(stderr,
                        "Error: -o '%s' is not a directory.\n", out_dir);
                return BAD_USAGE;
            }
            {
                int n = snprintf(out_prefix_buf, sizeof(out_prefix_buf),
                                 "%s/%s", out_dir, stem);
                if (n < 0 || n >= (int)sizeof(out_prefix_buf)) {
                    fprintf(stderr, "Error: output path too long.\n");
                    return BAD_USAGE;
                }
            }
            out_prefix = out_prefix_buf;
        }

        double gt_lo = 0.0, gt_hi = 0.0;
        int    gt_json_valid = 0;

        if (gt_path) {
            if (!is_tiff_path(gt_path)) {
                fprintf(stderr,
                        "Error: --ground must be a float32 .tif / .tiff file.\n");
                print_usage(argv[0]);
                return BAD_USAGE;
            }
            if (json_path) {
                if (parse_json_range(json_path, &gt_lo, &gt_hi) != 0) {
                    fprintf(stderr,
                            "Warning: could not parse JSON; "
                            "RMS still uses float32 TIFF truth.\n");
                } else {
                    gt_json_valid = 1;
                }
            }
        } else if (json_path) {
            fprintf(stderr,
                    "Warning: --json without --ground is ignored.\n");
        }

        elapsed_time = goldstein_phase_unwrapping(
            input_path, out_prefix,
            type,
            0, 0,
            mask_flag,
            gt_path, gt_lo, gt_hi, gt_json_valid, verify_serial,
            unwrap_backend);

    } else {
        /* ---- Default: built-in binary test data ---- */
        char data_path[PATH_MAX];
        char bin_input[PATH_MAX];
        char bin_prefix[PATH_MAX];
        int  MAX_ITERATIONS = 2;

        chdir("..");
        getcwd(data_path, sizeof(data_path));

        snprintf(bin_input,  sizeof(bin_input),
                 "%s/data/peaks.1024x1024.phase", data_path);
        snprintf(bin_prefix, sizeof(bin_prefix),
                 "%s/data/peaks.1024x1024",       data_path);

        for (i = 0; i < MAX_ITERATIONS; i++)
            elapsed_time += goldstein_phase_unwrapping(
                bin_input, bin_prefix,
                type, 1024, 1024, mask_flag,
                NULL, 0.0, 0.0, 0, verify_serial,
                unwrap_backend);

        elapsed_time /= (double)MAX_ITERATIONS;
        printf("\nAverage elapsed time: %f ms\n", elapsed_time);
    }

    return 0;
}
