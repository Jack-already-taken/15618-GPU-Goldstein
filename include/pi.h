#define PI              3.141592653589793
#define TWOPI           6.28318530717959

/* Tolerance for the 2x2 closed-loop phase-gradient sum when detecting
 * residues. Phase is in radians (one cycle = TWOPI); ideal |sum| is 0
 * or TWOPI; this is a small fraction of one cycle (same as 0.01 in old
 * normalised units). */
#define RESIDUE_THRESHOLD (0.01 * TWOPI)