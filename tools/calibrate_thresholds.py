# tools/calibrate_thresholds.py
import csv, json, argparse
from statistics import mean

def load_csv(path, key_col, val_col):
    # expected: first row header, columns include digits and median time columns
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            # try various column names for time (common names)
            digits = int(row.get('digits') or row.get('Digits') or row.get('digits_dec') or row.get('digits', 0))
            # pick the first time-like column
            t = None
            for k in row:
                if 'median' in k or 'median_s' in k or 'fft_median_s' in k or 'ntt_median_s' in k or 'median_time' in k:
                    try:
                        t = float(row[k])
                        break
                    except:
                        continue
            if t is None:
                # fallback: try any float value column
                for k in row:
                    try:
                        t = float(row[k])
                        break
                    except:
                        pass
            if t is None:
                continue
            rows.append((digits, t))
    return dict(sorted(rows))

def find_first_cross(a_map, b_map):
    # a_map and b_map: {digits: time}
    digits = sorted(set(a_map.keys()) & set(b_map.keys()))
    for d in digits:
        if a_map[d] < b_map[d]:
            return d
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fft', required=True)
    p.add_argument('--ntt', required=True)
    p.add_argument('--out', default='phase6/thresholds.json')
    args = p.parse_args()

    fft = load_csv(args.fft, 'digits', 'fft_median_s')
    ntt = load_csv(args.ntt, 'digits', 'ntt_median_s')

    # Attempt to find where FFT beats Karatsuba/Schoolbook and where NTT beats Schoolbook
    # Here we only have fft/ntt CSVs; we infer points roughly
    # Example heuristic: set FFT_PREFERRED to minimal digits where fft < ntt (or where fft << schoolbook)
    fft_vs_ntt = find_first_cross(fft, ntt)  # first d where fft < ntt
    # fallback reasonable defaults
    thresholds = {
        "SMALL_DIGITS": 100,
        "MEDIUM_DIGITS": 5000,
        "FFT_PREFERRED": fft_vs_ntt or 5000,
        "NTT_PREFERRED": max(100000, (fft_vs_ntt or 5000) * 10),
    }

    print("Derived thresholds:", thresholds)
    with open(args.out, 'w') as f:
        json.dump(thresholds, f, indent=2)
    print("Wrote thresholds to", args.out)

if __name__ == '__main__':
    main()
