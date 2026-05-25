"""
SEIDEL-2D Phase-4 GPU Pipeline (Option A: strategy NOT hardcoded) + MULTI-RUN (--runs N)

DIRECT : source only. Expected: ignores Gauss-Seidel loop-carried dependency
         (9-point, in-place) -> wrong (red-black/naive not order-preserving) or slow.
FULL   : cascade analysis drives generation. Expected: identifies dependency,
         picks wavefront -> but wavefront is unsound for this 9-point stencil
         (diagonal neighbor on same anti-diagonal) AND O(n) launches -> slow+wrong.

CORRECTNESS: CPU ref is strict serial Gauss-Seidel (unique result). A wavefront
that truly preserves GS order would match (~1e-10). A red-black / Jacobi-ized
kernel changes semantics -> large error: report it, don't call it a clean pass.

--runs N re-runs pipeline N times; each under results/cuda_seidel_pipeline/runK/.

Usage (WSL2, from project root, OPENAI_API_KEY + PYTHONUTF8=1):
    python3 test_cuda_seidel_full_pipeline.py --runs 3
"""
import sys, os, re, json, argparse, subprocess, logging
from datetime import datetime
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.insert(0, 'src')
from llm_client import LLMClient

N = 2000
TSTEPS = 20
NVCC_ARCH = "sm_86"
CORRECTNESS_TOL = 1e-6

SEIDEL_CODE = '''
void kernel_seidel_2d(int tsteps, int n, double A[n][n]) {
  for (int t = 0; t <= tsteps - 1; t++)
    for (int i = 1; i <= n - 2; i++)
      for (int j = 1; j <= n - 2; j++)
        A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
                 + A[i][j-1]   + A[i][j]   + A[i][j+1]
                 + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / 9.0;
}
'''

HOST_CONTRACT = '''
The generated code MUST expose exactly this host launcher (and any kernels it needs):

    void seidel_gpu(int tsteps, int n, double* A);

where:
  - A is a row-major n*n array of doubles already resident in DEVICE memory
    (allocated and initialized by the caller). Index element (i,j) as A[i*n + j].
  - seidel_gpu performs `tsteps` Gauss-Seidel sweeps over the interior
    1..n-2 x 1..n-2, updating A in place on the GPU, then returns.
  - Do NOT allocate/copy host memory inside seidel_gpu; operate on the device
    pointer A directly. You MAY allocate auxiliary device buffers if needed.
  - Output ONLY a single ```cuda code block containing kernels + seidel_gpu.
'''

BENCHMARK_TEMPLATE = r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(call) {{ cudaError_t e=call; if(e!=cudaSuccess){{printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);}} }}
#define N {N}
#define TSTEPS {TSTEPS}
// ===== LLM Generated (kernel + host launcher seidel_gpu) =====
{generated}
// ===== Strict serial CPU reference =====
static void seidel_cpu(int tsteps, int n, double* A) {{
    for (int t = 0; t <= tsteps - 1; t++)
        for (int i = 1; i <= n - 2; i++)
            for (int j = 1; j <= n - 2; j++)
                A[i*n + j] = (A[(i-1)*n + (j-1)] + A[(i-1)*n + j] + A[(i-1)*n + (j+1)]
                            + A[i*n + (j-1)]     + A[i*n + j]     + A[i*n + (j+1)]
                            + A[(i+1)*n + (j-1)] + A[(i+1)*n + j] + A[(i+1)*n + (j+1)]) / 9.0;
}}
static void init_array(int n, double* A, unsigned seed) {{
    srand(seed);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
        A[i*n + j] = (double)((i*(j+2) + 2) % 100) / n;
}}
static double max_abs_diff(const double* a, const double* b, int n) {{
    double m=0.0; for (int k=0;k<n*n;k++){{double e=fabs(a[k]-b[k]); if(e>m)m=e;}} return m;
}}
int main() {{
    int n=N, tsteps=TSTEPS; size_t bytes=(size_t)n*n*sizeof(double);
    double* h_cpu=(double*)malloc(bytes);
    double* h_gpu=(double*)malloc(bytes);
    double* h_A0 =(double*)malloc(bytes);
    init_array(n, h_A0, 12345u);
    double cpu_ms=0.0;
    for (int rep=0; rep<3; rep++) {{
        for (int k=0;k<n*n;k++) h_cpu[k]=h_A0[k];
        clock_t s=clock(); seidel_cpu(tsteps,n,h_cpu); clock_t e=clock();
        cpu_ms += (double)(e-s)/CLOCKS_PER_SEC*1000.0;
    }}
    cpu_ms/=3.0;
    double* d_A; CHECK_CUDA(cudaMalloc(&d_A,bytes));
    CHECK_CUDA(cudaMemcpy(d_A,h_A0,bytes,cudaMemcpyHostToDevice));
    seidel_gpu(tsteps,n,d_A);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_gpu,d_A,bytes,cudaMemcpyDeviceToHost));
    cudaEvent_t st,sp; cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    for (int rep=0; rep<5; rep++) {{
        CHECK_CUDA(cudaMemcpy(d_A,h_A0,bytes,cudaMemcpyHostToDevice));
        seidel_gpu(tsteps,n,d_A);
    }}
    cudaEventRecord(sp); cudaEventSynchronize(sp);
    float gpu_ms=0.0f; cudaEventElapsedTime(&gpu_ms,st,sp); gpu_ms/=5.0f;
    double err=max_abs_diff(h_cpu,h_gpu,n);
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms,gpu_ms,cpu_ms/(double)gpu_ms,err);
    cudaFree(d_A); free(h_cpu); free(h_gpu); free(h_A0);
    return 0;
}}
'''

def strategy_tag(code: str) -> str:
    c = code.lower(); tags = []
    if "diag" in c or "wavefront" in c or "anti-diagonal" in c or "antidiagonal" in c: tags.append("wavefront")
    if "red" in c and "black" in c: tags.append("red-black")
    if "color" in c or "colour" in c: tags.append("colored")
    if "__shared__" in c: tags.append("shared-mem")
    if "atomic" in c: tags.append("atomic")
    tags.append(f"{len(re.findall(r'__global__', code))}kernels")
    return "+".join(tags) if tags else "plain"

class SeidelPipeline:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.clients = {"gpt-4o": LLMClient(model="gpt-4o"), "gpt-5.2": LLMClient(model="gpt-5.2")}
        self.total_cost = 0.0
    def gen_direct(self) -> str:
        logger.info("[DIRECT] generating CUDA from source only...")
        p = f"\nTranslate the following C kernel into a CUDA GPU implementation.\n\n```c\n{SEIDEL_CODE}\n```\n\n{HOST_CONTRACT}\n"
        r = self.clients["gpt-5.2"].chat(p, system_prompt="Generate clean, working CUDA code.")
        self.total_cost += r.cost; return self._extract(r.content)
    def stage1(self) -> str:
        logger.info("[FULL/Stage1] GPT-4o analyzing kernel...")
        p = f'''
Analyze this HPC kernel for GPU parallelization:

```c
{SEIDEL_CODE}
```

Identify:
1. Performance bottleneck type (compute-bound or memory-bound) and why.
2. Any data dependencies that prevent straightforward parallelization.
3. A recommended GPU parallelization strategy.

Output JSON:
```json
{{"bottleneck_type":"...","data_dependencies":["..."],"parallelization_challenges":["..."],"recommended_strategy":"...","gpu_suitable":true}}
```
'''
        r = self.clients["gpt-4o"].chat(p, system_prompt="You are an HPC expert.")
        self.total_cost += r.cost; return r.content
    def stage2(self, s1: str) -> str:
        logger.info("[FULL/Stage2] GPT-5.2 validating + strategy...")
        p = f'''
Review the following analysis of a 2D stencil kernel and produce a CONCRETE GPU
parallelization strategy. First validate whether the previous bottleneck and
dependency analysis is correct; if it is wrong or incomplete, correct it.

## Previous analysis (Stage 1):
{s1}

## Original kernel:
```c
{SEIDEL_CODE}
```

Think carefully about whether the inner updates can be parallelized as-is, or
whether the update order imposes a dependency that a naive 2D thread mapping
would violate. Propose a specific, correct GPU scheme.

Output JSON:
```json
{{"validation":"correct/incorrect/partial","corrected_analysis":"...","gpu_strategy":{{"approach":"...","description":"...","preserves_semantics":true,"kernel_design":"..."}},"expected_speedup":"...","implementation_notes":["..."]}}
```
'''
        r = self.clients["gpt-5.2"].chat(p, system_prompt="You are a GPU parallelization expert.")
        self.total_cost += r.cost; return r.content
    def gen_full(self, s2: str) -> str:
        logger.info("[FULL/Stage3] generating CUDA driven by cascade strategy...")
        p = f'''
Using ONLY the parallelization strategy below, implement the seidel-2d kernel in CUDA.
Follow the strategy's chosen approach faithfully; do not substitute a different scheme.

## Strategy (from cascade analysis):
{s2}

## Original kernel (for reference):
```c
{SEIDEL_CODE}
```

{HOST_CONTRACT}
'''
        r = self.clients["gpt-5.2"].chat(p, system_prompt="Generate clean, working CUDA code.")
        self.total_cost += r.cost; return self._extract(r.content)
    @staticmethod
    def _extract(text: str) -> str:
        for pat in [r'```cuda\n(.*?)```', r'```cpp\n(.*?)```', r'```c\n(.*?)```', r'```\n(.*?)```']:
            m = re.search(pat, text, re.DOTALL)
            if m: return m.group(1).strip()
        return text.strip()
    def compile_and_run(self, generated: str, tag: str):
        src = BENCHMARK_TEMPLATE.format(generated=generated, N=N, TSTEPS=TSTEPS)
        cu = os.path.join(self.output_dir, f"seidel_{tag}.cu")
        binp = os.path.join(self.output_dir, f"seidel_{tag}")
        with open(cu, "w", encoding="utf-8") as f: f.write(src)
        logger.info(f"[{tag}] compiling {cu} ...")
        comp = subprocess.run(["nvcc","-O2",f"-arch={NVCC_ARCH}",cu,"-o",binp], capture_output=True, text=True)
        if comp.returncode != 0: return False, f"COMPILE_FAIL:\n{comp.stderr[-2000:]}", None
        logger.info(f"[{tag}] running ...")
        try: run = subprocess.run([binp], capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired: return False, "RUNTIME_TIMEOUT (>600s)", None
        out = run.stdout + "\n" + run.stderr
        m = re.search(r"BENCHMARK_RESULT:cpu_ms=([\d.]+),gpu_ms=([\d.]+),speedup=([\d.]+),error=([\d.eE+-]+)", out)
        if not m: return False, f"NO_RESULT_LINE:\n{out[-2000:]}", None
        res = {"cpu_ms": float(m.group(1)), "gpu_ms": float(m.group(2)), "speedup": float(m.group(3)), "error": float(m.group(4))}
        res["correct"] = res["error"] < CORRECTNESS_TOL
        return True, "OK", res
    def one_run(self) -> Dict[str, Any]:
        rep: Dict[str, Any] = {"kernel": "seidel_2d", "timestamp": datetime.now().isoformat()}
        dc = self.gen_direct()
        ok, msg, res = self.compile_and_run(dc, "direct")
        rep["direct"] = {"compiled_ran": ok, "msg": msg, "result": res, "strategy": strategy_tag(dc)}
        logger.info(f"[DIRECT] {msg} | {res} | {rep['direct']['strategy']}")
        s1 = self.stage1(); s2 = self.stage2(s1); fc = self.gen_full(s2)
        ok2, msg2, res2 = self.compile_and_run(fc, "full")
        rep["full"] = {"compiled_ran": ok2, "msg": msg2, "result": res2, "strategy": strategy_tag(fc), "stage1": s1, "stage2": s2}
        logger.info(f"[FULL] {msg2} | {res2} | {rep['full']['strategy']}")
        rep["total_cost"] = self.total_cost
        with open(os.path.join(self.output_dir, "seidel_pipeline_report.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        return rep

def fmt(rep, which):
    r = rep[which]
    if r["result"]:
        rr = r["result"]
        return f"speedup={rr['speedup']:7.2f}x  err={rr['error']:.1e}  correct={str(rr['correct']):5}  [{r['strategy']}]"
    return f"FAILED: {r['msg'].splitlines()[0]:40} [{r.get('strategy','?')}]"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    base = "results/cuda_seidel_pipeline"; reps: List[Dict[str, Any]] = []
    for k in range(1, args.runs + 1):
        outdir = base if args.runs == 1 else f"{base}/run{k}"
        logger.info(f"\n########## SEIDEL RUN {k}/{args.runs} -> {outdir} ##########")
        reps.append(SeidelPipeline(outdir).one_run())
    print("\n" + "="*78); print(f"SEIDEL-2D Phase-4 stability summary ({args.runs} run(s))"); print("="*78)
    for i, rep in enumerate(reps, 1):
        print(f"--- run {i} ---")
        print(f"  DIRECT  {fmt(rep,'direct')}")
        print(f"  FULL    {fmt(rep,'full')}")
    if args.runs > 1:
        fs = [r['full']['result']['speedup'] for r in reps if r['full']['result']]
        fc = [r['full']['result']['correct'] for r in reps if r['full']['result']]
        strat = set(r['full']['strategy'] for r in reps)
        print("-"*78)
        if fs: print(f"  FULL speedup range: {min(fs):.2f}x .. {max(fs):.2f}x")
        print(f"  FULL strategies seen: {strat}")
        print(f"  FULL correct in any run: {any(fc)} ; correct in all: {all(fc) if fc else False}")
        print(f"  Slowdown reproducible (FULL slower than CPU, <1x, every run): {all(s < 1.0 for s in fs) if fs else 'n/a'}")

if __name__ == "__main__":
    main()
