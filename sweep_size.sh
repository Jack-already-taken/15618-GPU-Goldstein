#!/usr/bin/env bash
#
# sweep_size.sh — size scaling sweep across serial CPU and CUDA backends.
#
# Runs src/main on a series of size_{N}x{N}_wrapped.tif inputs, parses the
# per-stage timing report printed to stdout, and writes one CSV row per
# (size, backend) pair. Designed to be re-run cheaply: the CSV is overwritten
# on each invocation, and the output TIFFs land in a dedicated dir.
#
# Usage:
#   ./sweep_size.sh                              # defaults
#   ./sweep_size.sh -r 3 -o results.csv          # 3 repeats, custom CSV
#   ./sweep_size.sh -s "256 512 1024" -b "serial cuda"
#
# Output CSV columns:
#   size,backend,repeat,num_residues,num_pieces,
#   load_ms,init_bitflags_ms,gradxy_ms,cuda_init_ms,
#   residues_ms,branch_cuts_ms,unwrap_ms,kernel_total_ms,
#   write_ms,total_ms

set -euo pipefail

# ---- defaults ---------------------------------------------------------------
SIZES=(256 512 1024 2048 4096 8192)
BACKENDS=(serial cuda)
REPEATS=1
PHASE_DIR="phase_data/size_scaling"
OUT_DIR="sweep_out"
CSV="sweep_results.csv"
BIN="src/main"
THREADS=""     # empty -> let main.c pick its default

# ---- args -------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 [options]
  -s "256 512 ..."    space-separated sizes (default: ${SIZES[*]})
  -b "serial cuda"    space-separated backends (default: ${BACKENDS[*]})
  -r N                repeats per (size, backend) pair (default: $REPEATS)
  -d DIR              phase data dir (default: $PHASE_DIR)
  -o FILE             output CSV (default: $CSV)
  -O DIR              output TIFF dir (default: $OUT_DIR)
  -B PATH             binary path (default: $BIN)
  -t N                OpenMP threads (default: main.c's own default)
  -h                  show this help
EOF
    exit 0
}

while getopts "s:b:r:d:o:O:B:t:h" opt; do
    case "$opt" in
        s) read -r -a SIZES    <<< "$OPTARG" ;;
        b) read -r -a BACKENDS <<< "$OPTARG" ;;
        r) REPEATS="$OPTARG" ;;
        d) PHASE_DIR="$OPTARG" ;;
        o) CSV="$OPTARG" ;;
        O) OUT_DIR="$OPTARG" ;;
        B) BIN="$OPTARG" ;;
        t) THREADS="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ ! -x "$BIN" ]]; then
    echo "error: binary '$BIN' not found or not executable" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# ---- CSV header -------------------------------------------------------------
echo "size,backend,repeat,num_residues,num_pieces,load_ms,init_bitflags_ms,gradxy_ms,cuda_init_ms,residues_ms,branch_cuts_ms,unwrap_ms,kernel_total_ms,write_ms,total_ms" > "$CSV"

# ---- parser ------------------------------------------------------------------
# Given the captured stdout of one main run, emit:
#   num_residues,num_pieces,load,init_bf,gradxy,cuda_init,res,bc,unwrap,ksub,write
# one field per line (11 fields). Empty string if a field isn't present.
parse_run() {
    awk '
    BEGIN {
        for (i = 0; i < 11; i++) f[i] = ""
    }
    /^Number of residues:/      { f[0]  = $NF }
    /^Number of pieces:/        { f[1]  = $NF }
    /Load wrapped phase/        { f[2]  = $NF }
    /Init bitflags/             { f[3]  = $NF }
    /Gradxy/                    { f[4]  = $NF }
    /CUDA init \+ device malloc/{ f[5]  = $NF }
    /^ {4}Residues /            { f[6]  = $NF }
    /^ {4}Branch cuts /         { f[7]  = $NF }
    /^ {4}Unwrap /              { f[8]  = $NF }
    /Kernel subtotal/           { f[9]  = $NF }
    /Write output TIFF/         { f[10] = $NF }
    END {
        for (i = 0; i < 11; i++) print f[i]
    }'
}

# ---- main loop --------------------------------------------------------------
run_one() {
    local size="$1" backend="$2" rep="$3"
    local input="${PHASE_DIR}/size_${size}x${size}_wrapped.tif"

    if [[ ! -f "$input" ]]; then
        echo "skip: missing $input" >&2
        return
    fi

    local cmd=("$BIN" -i "$input" -o "$OUT_DIR" -B "$backend")
    [[ -n "$THREADS" ]] && cmd+=(-t "$THREADS")

    echo "[run] size=${size} backend=${backend} repeat=${rep}" >&2
    local out
    if ! out=$("${cmd[@]}" 2>&1); then
        echo "warn: run failed for size=${size} backend=${backend}" >&2
        echo "$out" >&2
        return
    fi

    # Parse into 11 fields
    mapfile -t fields < <(echo "$out" | parse_run)
    local num_res="${fields[0]}"
    local num_pie="${fields[1]}"
    local t_load="${fields[2]}"
    local t_init="${fields[3]}"
    local t_grad="${fields[4]}"
    local t_cuda="${fields[5]}"
    local t_res="${fields[6]}"
    local t_bc="${fields[7]}"
    local t_unw="${fields[8]}"
    local t_ksub="${fields[9]}"
    local t_wr="${fields[10]}"

    # total = sum of available timed stages
    local total
    total=$(awk -v a="$t_load" -v b="$t_init" -v c="$t_grad" -v d="$t_cuda" \
                -v e="$t_ksub" -v f="$t_wr" \
        'BEGIN {
            t = 0
            if (a != "") t += a
            if (b != "") t += b
            if (c != "") t += c
            if (d != "") t += d
            if (e != "") t += e
            if (f != "") t += f
            printf "%.3f", t
        }')

    echo "${size},${backend},${rep},${num_res},${num_pie},${t_load},${t_init},${t_grad},${t_cuda},${t_res},${t_bc},${t_unw},${t_ksub},${t_wr},${total}" >> "$CSV"
}

for size in "${SIZES[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        for ((r = 1; r <= REPEATS; r++)); do
            run_one "$size" "$backend" "$r"
        done
    done
done

echo "done. wrote $CSV"
