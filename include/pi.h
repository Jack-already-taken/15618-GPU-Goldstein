#define PI              3.141592653589793
#define TWOPI           6.28318530717959

/* Tolerance for the 2x2 closed-loop phase-gradient sum when detecting
 * residues. In normalised units; the ideal sum is exactly 0 or +/-1
 * (i.e. 0 or +/-2*PI radians), so this is a small epsilon around 0. */
#define RESIDUE_THRESHOLD 0.01