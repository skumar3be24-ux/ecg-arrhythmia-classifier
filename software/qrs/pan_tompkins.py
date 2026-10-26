"""
pan_tompkins.py — QRS detection, float and bit-exact fixed-point.

The fixed-point version is the GOLDEN MODEL for the Verilog. Every operation
in `pan_tompkins_fixed` is one a Verilog always-block can perform: integer
add/subtract, arithmetic shift, one multiply (the squarer), and comparison.
No floats, no division.

Algorithm: Pan, J. & Tompkins, W. J., "A Real-Time QRS Detection Algorithm",
IEEE Trans. Biomedical Engineering, BME-32(3):230-236, 1985.
DOI 10.1109/TBME.1985.325532
Filter equations from the published errata.

Original design rate is 200 Hz. We run at 180 Hz (MIT-BIH 360 Hz decimated by
2). Cutoffs shift 11 Hz -> 9.9 Hz and 5 Hz -> 4.5 Hz, both still inside the
QRS energy band. [MUST MEASURE] the sensitivity cost — evaluate_qrs.py does
exactly that.
"""

import numpy as np

FS = 180                       # Hz, after decimation by 2
REFRACTORY   = 36              # 200 ms  @ 180 Hz
TWAVE_WIN    = 64              # 360 ms  @ 180 Hz
MWI_N        = 32              # 178 ms window; power of 2 so /N is a shift
MWI_SHIFT    = 5               # log2(32)
SEARCH_RADIUS = 8              # +/-44 ms when locating the true R peak

# Group delay of the filter chain, in samples.
#
# RE-CALIBRATED 18-Aug-2026 after the halfband decimator bug was fixed.
# The old value (28) had been fitted to the BROKEN decimator; with the correct
# filter the optimum moved to 36 -- much closer to the theoretical sum
# (LP 6 + HP 16 + deriv 2 + MWI_N/2 16 = 40), which is itself evidence the
# decimator was distorting the signal.
#
# Swept group_delay 8..38 x radius 8/12/16 on an 11-record DS1 subset
# (DS2 NEVER used for calibration). Broad plateau Se~98.7% PPV~99.0% for
# group_delay 30..38 at radius 8. 34 sits mid-plateau.
GROUP_DELAY = 34


# ===========================================================================
#  FIXED-POINT HELPERS  (mirror these exactly in Verilog)
# ===========================================================================

def sat(x, bits):
    """Saturate to a signed `bits`-wide value. ALWAYS saturate, never wrap."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return int(np.clip(x, lo, hi))


def asr(x, n):
    """Arithmetic right shift. Python's >> is already arithmetic for ints."""
    return x >> n


# ===========================================================================
#  DECIMATION 360 -> 180 Hz
# ===========================================================================

# 11-tap halfband FIR, coefficients scaled by 64 so the divide is >> 6.
# Zero-valued taps are the halfband property; they cost nothing in hardware.
#
# BUG FIXED 18-Aug-2026: the previous coefficients were
#     [-2, 0, 6, 0, -19, 32, -19, 0, 6, 0, -2]
# which sum to 2, not 64 — a DC gain of 0.031, attenuating the signal 32x, and
# with NEGATIVE neighbours around the centre tap it was not a lowpass at all.
# Pan-Tompkins is amplitude-adaptive so it still worked, which is exactly why
# the bug survived undetected until the vector range check exposed it
# (decimated signal range was -8..+10 where ADC units should be ~+/-300).
#
# Verified: sum = 64 -> DC gain 1.0; |H| = 0.53 at fs/4; 0.0625 at fs/2.
HALFBAND = [0, 0, -3, 0, 18, 34, 18, 0, -3, 0, 0]
HALFBAND_SHIFT = 6
assert sum(HALFBAND) == (1 << HALFBAND_SHIFT), "halfband DC gain must be 1.0"


def decimate_fixed(x):
    """Integer 2:1 decimator. x: int array at 360 Hz -> int array at 180 Hz."""
    x = np.asarray(x, dtype=np.int64)
    pad = len(HALFBAND) // 2
    xp = np.concatenate([np.zeros(pad, np.int64), x, np.zeros(pad, np.int64)])
    out = []
    for n in range(0, len(x), 2):
        acc = 0
        for k, c in enumerate(HALFBAND):
            acc += c * int(xp[n + k])
        out.append(asr(acc, HALFBAND_SHIFT))
    return np.array(out, dtype=np.int64)


# ===========================================================================
#  PAN-TOMPKINS FILTER STAGES  (integer, no multipliers except the squarer)
# ===========================================================================

def lowpass_fixed(x):
    """y[n] = 2y[n-1] - y[n-2] + x[n] - 2x[n-6] + x[n-12]     (gain 36)"""
    N = len(x)
    y = np.zeros(N, dtype=np.int64)
    for n in range(N):
        acc = int(x[n])
        if n >= 6:  acc -= 2 * int(x[n - 6])
        if n >= 12: acc += int(x[n - 12])
        if n >= 1:  acc += 2 * int(y[n - 1])
        if n >= 2:  acc -= int(y[n - 2])
        y[n] = acc
    return y


def highpass_fixed(x):
    """y[n] = 32*x[n-16] - (y[n-1] + x[n] - x[n-32])          (unity gain)"""
    N = len(x)
    y = np.zeros(N, dtype=np.int64)
    for n in range(N):
        acc = 0
        if n >= 16: acc += int(x[n - 16]) << 5      # *32 is a shift
        acc -= int(x[n])
        if n >= 32: acc += int(x[n - 32])
        if n >= 1:  acc -= int(y[n - 1])
        y[n] = acc
    return y


def derivative_fixed(x):
    """y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) >> 3        (gain 0.1)"""
    N = len(x)
    y = np.zeros(N, dtype=np.int64)
    for n in range(N):
        acc = 2 * int(x[n])
        if n >= 1: acc += int(x[n - 1])
        if n >= 3: acc -= int(x[n - 3])
        if n >= 4: acc -= 2 * int(x[n - 4])
        y[n] = asr(acc, 3)
    return y


def square_fixed(x):
    """y[n] = x[n] * x[n]  — THE ONLY MULTIPLIER IN THE WHOLE FRONT-END."""
    return (np.asarray(x, dtype=np.int64) ** 2)


def mwi_fixed(x, N=MWI_N, shift=MWI_SHIFT):
    """Moving-window integration via a running sum: 2 adds per sample."""
    out = np.zeros(len(x), dtype=np.int64)
    run = 0
    for n in range(len(x)):
        run += int(x[n])
        if n >= N:
            run -= int(x[n - N])
        out[n] = asr(run, shift)
    return out


# ===========================================================================
#  ADAPTIVE THRESHOLD + DECISION
# ===========================================================================

# Ablation flags. Defaults chosen by MEASUREMENT over all 44 records, not
# by theory. See archive/results/qrs_ablation.md.
USE_SEARCHBACK      = False
USE_DUAL_THRESHOLD  = False


def detect_peaks_fixed(integrated, bandpass, fs=FS, searchback=None, dual=None):
    """
    Full Pan-Tompkins decision logic, integer-only.

    Implements ALL of:
      - dual adaptive thresholds (integrated signal AND bandpass signal)
      - refractory blanking
      - T-wave discrimination
      - search-back on 166% of the average RR
      - 8-beat RR averaging

    The dual threshold is what suppresses false positives; search-back is what
    recovers beats after the threshold over-adapts to a large peak. Omitting
    either costs several percent — measured, see tools/ablation_qrs.py.

    All coefficients are powers of two:
        0.25  -> >> 2      0.125 -> >> 3      0.875 -> x - (x >> 3)
        0.5   -> >> 1      1.66  ~ x + (x>>1) + (x>>4)  = 1.5625x
    """
    N = len(integrated)
    learn = min(2 * fs, N)
    SPKI = int(np.max(integrated[:learn])) >> 1
    NPKI = int(np.mean(integrated[:learn])) >> 1
    absbp = np.abs(bandpass).astype(np.int64)
    SPKF = int(np.max(absbp[:learn])) >> 1
    NPKF = int(np.mean(absbp[:learn])) >> 1

    # 1. collect every local maximum of the integrated signal, once.
    cand = [n for n in range(1, N - 1)
            if integrated[n] > integrated[n - 1]
            and integrated[n] >= integrated[n + 1]]

    peaks, rr8 = [], []
    last_i = -REFRACTORY - 1
    last_slope = 0
    rr_avg = 0

    def locate(n):
        """Undo the group delay and find the true R peak in the bandpass."""
        c = n - GROUP_DELAY
        a = max(0, c - SEARCH_RADIUS)
        b = min(len(bandpass), c + SEARCH_RADIUS + 1)
        if b <= a:
            return max(0, c)
        return a + int(np.argmax(absbp[a:b]))

    def accept(n, half=False):
        nonlocal SPKI, NPKI, SPKF, NPKF, last_i, last_slope, rr_avg
        pk = int(integrated[n])
        # search-back detections update the estimates more gently (1/4 not 1/8)
        SPKI = asr(pk, 2) + (SPKI - asr(SPKI, 2)) if half else \
               asr(pk, 3) + (SPKI - asr(SPKI, 3))
        r = locate(n)
        pf = int(absbp[min(r, len(absbp) - 1)])
        SPKF = asr(pf, 2) + (SPKF - asr(SPKF, 2)) if half else \
               asr(pf, 3) + (SPKF - asr(SPKF, 3))
        lo = max(0, n - 6)
        last_slope = abs(int(integrated[n]) - int(integrated[lo]))
        if peaks:
            rr = r - peaks[-1]
            if rr > 0:
                rr8.append(rr)
                if len(rr8) > 8:
                    rr8.pop(0)
                rr_avg = sum(rr8) // len(rr8)
        peaks.append(r)
        last_i = n
        return r

    dl = USE_DUAL_THRESHOLD if dual is None else dual

    for idx, n in enumerate(cand):
        pk = int(integrated[n])
        thr_i = NPKI + asr(SPKI - NPKI, 2)

        # ---- search-back: no beat for 166% of the average RR? -----------
        sb = USE_SEARCHBACK if searchback is None else searchback
        dl = USE_DUAL_THRESHOLD if dual is None else dual
        if sb and rr_avg > 0 and peaks:
            limit = rr_avg + asr(rr_avg, 1) + asr(rr_avg, 4)   # ~1.5625x
            if (n - GROUP_DELAY) - peaks[-1] > limit:
                half_i = NPKI + asr(SPKI - NPKI, 3)
                half_f = NPKF + asr(SPKF - NPKF, 3)
                best, bv = -1, 0
                for m in cand:
                    if not (last_i < m < n):
                        continue
                    if m - last_i < REFRACTORY:
                        continue
                    v = int(integrated[m])
                    if v <= half_i or v <= bv:
                        continue
                    # a search-back candidate must ALSO clear the bandpass
                    # threshold, or noise peaks get promoted to beats
                    rr_ = locate(m)
                    if int(absbp[min(rr_, len(absbp) - 1)]) <= half_f:
                        continue
                    best, bv = m, v
                if best >= 0:
                    accept(best, half=True)

        if pk <= thr_i:
            NPKI = asr(pk, 3) + (NPKI - asr(NPKI, 3))
            continue
        if n - last_i < REFRACTORY:
            continue

        # ---- SECOND THRESHOLD on the bandpass signal --------------------
        r = locate(n)
        pf = int(absbp[min(r, len(absbp) - 1)])
        thr_f = NPKF + asr(SPKF - NPKF, 2)
        if dl and pf <= thr_f:
            NPKI = asr(pk, 3) + (NPKI - asr(NPKI, 3))
            NPKF = asr(pf, 3) + (NPKF - asr(NPKF, 3))
            continue

        # ---- T-wave discrimination --------------------------------------
        if peaks and (n - last_i) < TWAVE_WIN:
            lo = max(0, n - 6)
            slope = abs(int(integrated[n]) - int(integrated[lo]))
            if slope < asr(last_slope, 1):
                NPKI = asr(pk, 3) + (NPKI - asr(NPKI, 3))
                NPKF = asr(pf, 3) + (NPKF - asr(NPKF, 3))
                continue

        accept(n)

    peaks = np.array(sorted(set(peaks)), dtype=np.int64)
    return peaks, np.diff(peaks) if len(peaks) > 1 else np.array([], np.int64)


# ===========================================================================
#  TOP-LEVEL
# ===========================================================================

def pan_tompkins_fixed(x360, return_stages=False, searchback=None, dual=None):
    """
    Full integer pipeline. Input: raw ADC integers at 360 Hz.
    Output: R-peak indices at 180 Hz.
    """
    x = decimate_fixed(x360)
    lp = lowpass_fixed(x)
    hp = highpass_fixed(lp)
    dv = derivative_fixed(hp)
    sq = square_fixed(dv)
    mw = mwi_fixed(sq)
    peaks, rr = detect_peaks_fixed(mw, hp, searchback=searchback, dual=dual)
    if return_stages:
        return peaks, rr, dict(dec=x, lp=lp, hp=hp, deriv=dv, sq=sq, mwi=mw)
    return peaks, rr


def pan_tompkins_float(x360, fs_in=360):
    """Floating-point reference (scipy). Used only to sanity-check the
    fixed-point version — the Verilog follows the FIXED version."""
    from scipy.signal import butter, filtfilt, decimate
    x = decimate(np.asarray(x360, float), 2, ftype='fir', zero_phase=True)
    b, a = butter(2, [5.0 / (FS / 2), 15.0 / (FS / 2)], btype='band')
    bp = filtfilt(b, a, x)
    dv = np.convolve(bp, np.array([2, 1, 0, -1, -2]) / 8.0, mode='same')
    sq = dv ** 2
    mw = np.convolve(sq, np.ones(MWI_N) / MWI_N, mode='same')
    peaks, _ = detect_peaks_fixed((mw * 1e6).astype(np.int64),
                                 (bp * 1000).astype(np.int64))
    return peaks
