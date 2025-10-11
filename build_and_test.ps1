# build_and_test.ps1
param(
    [int]$Jobs = 8,
    [switch]$RunBench,
    [switch]$CleanOnly
)

Set-StrictMode -Version Latest
Write-Host "PWD:" (Get-Location)
Write-Host "Jobs:" $Jobs "RunBench:" $RunBench "CleanOnly:" $CleanOnly

# 1) Clean
Write-Host "`n--- Cleaning build artifacts ---"
Remove-Item -Recurse -Force .\build, .\dist -ErrorAction SilentlyContinue
Remove-Item -Force .\src\myext\*.pyd -ErrorAction SilentlyContinue
Remove-Item -Force .\fast_gemm\*.pyd -ErrorAction SilentlyContinue

if ($CleanOnly) {
    Write-Host "Clean-only requested — done."
    exit 0
}

# 2) Ensure build deps (pip output saved to log)
Write-Host "`n--- Ensuring build tools (pip install) ---"
python -m pip install --upgrade pip setuptools wheel cython numpy | Tee-Object -FilePath .\pip_buildtools.log

# 3) Build extension (uses setup_myext.py in repo root)
Write-Host "`n--- Building myext (setup_myext.py) ---"
$buildCmd = "python .\setup_myext.py build_ext --inplace -j $Jobs"
Write-Host "CMD: $buildCmd"
# Run and capture output
Invoke-Expression $buildCmd 2>&1 | Tee-Object -FilePath .\build_myext_log.txt
Write-Host "`n--- Last 40 lines of build log ---"
if (Test-Path .\build_myext_log.txt) { Get-Content .\build_myext_log.txt -Tail 40 } else { Write-Host "No build log found." }

# 4) Copy built .pyd into src\myext if found under build
Write-Host "`n--- Copy built .pyd (if present) ---"
$pyd = Get-ChildItem -Path .\build -Recurse -Include *gemm*.pyd -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($pyd) {
    Copy-Item $pyd.FullName -Destination .\src\myext\ -Force
    Write-Host "Copied" $pyd.Name "to src\myext"
} else {
    Write-Host "No gemm .pyd under build; maybe already in src\myext or build failed."
}

# 5) Run pytest and save output
Write-Host "`n--- Running pytest ---"
python -m pytest -q 2>&1 | Tee-Object -FilePath .\pytest_run.log
Write-Host "`n--- pytest last 40 lines ---"
if (Test-Path .\pytest_run.log) { Get-Content .\pytest_run.log -Tail 40 } else { Write-Host "No pytest log found." }

# 6) Optional micro-benchmark (creates temp script and runs it)
if ($RunBench) {
    Write-Host "`n--- Running micro-benchmark (256x256) ---"
    $bench_py = @'
import time, numpy as np, sys
sys.path.insert(0, "src")
import myext
A = np.random.rand(256,256)
B = np.random.rand(256,256)
t0 = time.perf_counter()
myext.gemm(A,B)
t1 = time.perf_counter()
print(f"myext.gemm: {t1-t0:.6f}s")
t0 = time.perf_counter()
np.dot(A,B)
t1 = time.perf_counter()
print(f"numpy.dot  : {t1-t0:.6f}s")
'@
    $temp = ".\bench_temp.py"
    $bench_py | Set-Content -Encoding UTF8 $temp
    python $temp
    Remove-Item $temp -ErrorAction SilentlyContinue
}

Write-Host "`nDone ✅"
