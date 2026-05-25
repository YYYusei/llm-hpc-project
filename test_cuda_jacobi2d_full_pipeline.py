"""
JACOBI-2D Phase-4 GPU Pipeline (Option A: strategy NOT hardcoded) + MULTI-RUN (--runs N)

NEUTRAL control. Jacobi-2d = 5-point stencil, DOUBLE-BUFFERED (A<->B): no
loop-carried dependency, embarrassingly parallel. Expected: DIRECT and FULL both
correct and similar speed -> cascade neither necessary nor harmful.

No semantics trap (serial == parallel Jacobi), so correct kernel -> error ~1e-13.

--runs N re-runs pipeline N times; each under results/cuda_jacobi2d_pipeline/runK/.

Usage (WSL2, from project root, OPENAI_API_KEY + PYTHONUTF8=1):
    python3 test_cuda_jacobi2d_full_pipeline.py --runs 3
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

JACOBI_CODE = '''
void kernel_jacobi_2d(int tsteps, int n, double A[n][n], double B[n][n]) {
  for (int t = 0; t < tsteps; t++) {
    for (int i = 1; i < n - 1; i++)
      for (int j = 1; j < n - 1; j++)
        B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
    for (int i = 1; i < n - 1; i++)
      for (int j = 1; j < n - 1; j++)
        A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
  }
}
'''

HOST_CONTRACT = '''
The generated code MUST expose exactly this host launcher (plus any kernels):

    void jacobi_gpu(int tsteps, int n, double* A, double* B);

where (both DEVICE pointers, row-major n*n, allocated/initialized by caller):
  - One time step does: B = stencil(A) on interior 1..n-2, then A = stencil(B).
    Repeat tsteps times. After return, the final result must be in A (matching
    the serial reference, which ends with A updated from B).
  - Index element (i,j) as ptr[i*n + j]. Do NOT copy to/from host inside
    jacobi_gpu; operate on device pointers directly.
  - Output ONLY a single ```cuda code block.
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
// ===== LLM Generated (kernels + host launcher jacobi_gpu) =====
{generated}
// ===== Strict serial CPU reference =====
static void jacobi_cpu(int tsteps, int n, double* A, double* B) {{
    for (int t = 0; t < tsteps; t++) {{
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                B[i*n+j] = 0.2 * (A[i*n+j] + A[i*n+(j-1)] + A[i*n+(1+j)] + A[(1+i)*n+j] + A[(i-1)*n+j]);
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                A[i*n+j] = 0.2 * (B[i*n+j] + B[i*n+(j-1)] + B[i*n+(1+j)] + B[(1+i)*n+j] + B[(i-1)*n+j]);
    }}
}}
static void init_arrays(int n, double* A, double* B) {{
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) {{
        A[i*n+j]=((double)i*(j+2)+2)/n; B[i*n+j]=((double)i*(j+3)+3)/n;
    }}
}}
static double max_abs_diff(const double* a, const double* b, int n) {{
    double m=0.0; for (int k=0;k<n*n;k++){{double e=fabs(a[k]-b[k]); if(e>m)m=e;}} return m;
}}
int main() {{
    int n=N, tsteps=TSTEPS; size_t bytes=(size_t)n*n*sizeof(double);
    double* A0=(double*)malloc(bytes);
    double* B0=(double*)malloc(bytes);
    double* Ac=(double*)malloc(bytes);
    double* Bc=(double*)malloc(bytes);
    double* Ag=(double*)malloc(bytes);
    init_arrays(n, A0, B0);
    double cpu_ms=0.0;
    for (int rep=0; rep<3; rep++) {{
        for (size_t k=0;k<(size_t)n*n;k++){{ Ac[k]=A0[k]; Bc[k]=B0[k]; }}
        clock_t s=clock(); jacobi_cpu(tsteps,n,Ac,Bc); clock_t e=clock();
        cpu_ms += (double)(e-s)/CLOCKS_PER_SEC*1000.0;
    }}
    cpu_ms/=3.0;
    double *dA,*dB; CHECK_CUDA(cudaMalloc(&dA,bytes)); CHECK_CUDA(cudaMalloc(&dB,bytes));
    CHECK_CUDA(cudaMemcpy(dA,A0,bytes,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,B0,bytes,cudaMemcpyHostToDevice));
    jacobi_gpu(tsteps,n,dA,dB);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Ag,dA,bytes,cudaMemcpyDeviceToHost));
    cudaEvent_t st,sp; cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    for (int rep=0; rep<5; rep++) {{
        CHECK_CUDA(cudaMemcpy(dA,A0,bytes,cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB,B0,bytes,cudaMemcpyHostToDevice));
        jacobi_gpu(tsteps,n,dA,dB);
    }}
    cudaEventRecord(sp); cudaEventSynchronize(sp);
    float gpu_ms=0.0f; cudaEventElapsedTime(&gpu_ms,st,sp); gpu_ms/=5.0f;
    double err=max_abs_diff(Ac,Ag,n);
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms,gpu_ms,cpu_ms/(double)gpu_ms,err);
    cudaFree(dA); cudaFree(dB);
    free(A0); free(B0); free(Ac); free(Bc); free(Ag);
    return 0;
}}
'''

def strategy_tag(code: str) -> str:
    c = code.lower(); tags = []
    if "__shared__" in c: tags.append("shared-mem")
    if "fused" in c: tags.append("fused")
    tags.append(f"{len(re.findall(r'__global__', code))}kernels")
    return "+".join(tags) if tags else "plain"

class JacobiPipeline:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.clients = {"gpt-4o": LLMClient(model="gpt-4o"), "gpt-5.2": LLMClient(model="gpt-5.2")}
        self.total_cost = 0.0
    def gen_direct(self) -> str:
        logger.info("[DIRECT] generating CUDA from source only...")
        p = f"\nTranslate the following C kernel into a CUDA GPU implementation.\n\n```c\n{JACOBI_CODE}\n```\n\n{HOST_CONTRACT}\n"
        r = self.clients["gpt-5.2"].chat(p, system_prompt="Generate clean, working CUDA code.")
        self.total_cost += r.cost; return self._extract(r.content)
    def stage1(self) -> str:
        logger.info("[FULL/Stage1] GPT-4o analyzing...")
        p = f'''
Analyze this HPC kernel for GPU parallelization:

```c
{JACOBI_CODE}
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
Review the following analysis of a 2D Jacobi stencil kernel and produce a CONCRETE
GPU parallelization strategy. First validate whether the previous bottleneck and
dependency analysis is correct; if wrong or incomplete, correct it.

## Previous analysis (Stage 1):
{s1}

## Original kernel:
```c
{JACOBI_CODE}
```

Output JSON:
```json
{{"validation":"correct/incorrect/partial","corrected_analysis":"...","gpu_strategy":{{"approach":"...","description":"...","kernel_design":"..."}},"expected_speedup":"...","implementation_notes":["..."]}}
```
'''
        r = self.clients["gpt-5.2"].chat(p, system_prompt="You are a GPU parallelization expert.")
        self.total_cost += r.cost; return r.content
    def gen_full(self, s2: str) -> str:
        logger.info("[FULL/Stage3] generating CUDA driven by cascade strategy...")
        p = f'''
Using ONLY the parallelization strategy below, implement the jacobi-2d kernel in CUDA.
Follow the strategy's chosen approach faithfully; do not substitute a different scheme.

## Strategy (from cascade analysis):
{s2}

## Original kernel (for reference):
```c
{JACOBI_CODE}
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
        cu = os.path.join(self.output_dir, f"jacobi2d_{tag}.cu")
        binp = os.path.join(self.output_dir, f"jacobi2d_{tag}")
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
        rep: Dict[str, Any] = {"kernel": "jacobi2d", "timestamp": datetime.now().isoformat()}
        dc = self.gen_direct()
        ok, msg, res = self.compile_and_run(dc, "direct")
        rep["direct"] = {"compiled_ran": ok, "msg": msg, "result": res, "strategy": strategy_tag(dc)}
        logger.info(f"[DIRECT] {msg} | {res} | {rep['direct']['strategy']}")
        s1 = self.stage1(); s2 = self.stage2(s1); fc = self.gen_full(s2)
        ok2, msg2, res2 = self.compile_and_run(fc, "full")
        rep["full"] = {"compiled_ran": ok2, "msg": msg2, "result": res2, "strategy": strategy_tag(fc), "stage1": s1, "stage2": s2}
        logger.info(f"[FULL] {msg2} | {res2} | {rep['full']['strategy']}")
        rep["total_cost"] = self.total_cost
        with open(os.path.join(self.output_dir, "jacobi2d_pipeline_report.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        return rep

def fmt(rep, which):
    r = rep[which]
    if r["result"]:
        rr = r["result"]
        return f"speedup={rr['speedup']:7.2f}x  err={rr['error']:.1e}  correct={str(rr['correct']):5}  [{r['strategy']}]"
    return f"FAILED: {r['msg'].splitlines()[0]:40} [{r.get('strategy','?')}]"

# classify a FULL run relative to its DIRECT baseline (fixes the misleading "<" verdict)
def classify(full_sp, direct_sp):
    if full_sp is None: return "fail"
    if full_sp < 1.0: return "catastrophic(<1x)"          # slower than CPU = clear over-correction
    if full_sp < 0.5 * direct_sp: return "degraded(<0.5x DIRECT)"
    return "comparable"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    base = "results/cuda_jacobi2d_pipeline"; reps: List[Dict[str, Any]] = []
    for k in range(1, args.runs + 1):
        outdir = base if args.runs == 1 else f"{base}/run{k}"
        logger.info(f"\n########## JACOBI2D RUN {k}/{args.runs} -> {outdir} ##########")
        reps.append(JacobiPipeline(outdir).one_run())
    print("\n" + "="*78); print(f"JACOBI-2D Phase-4 stability summary ({args.runs} run(s))"); print("="*78)
    for i, rep in enumerate(reps, 1):
        dsp = rep['direct']['result']['speedup'] if rep['direct']['result'] else None
        fsp = rep['full']['result']['speedup'] if rep['full']['result'] else None
        cls = classify(fsp, dsp) if (dsp is not None) else "?"
        print(f"--- run {i} ---")
        print(f"  DIRECT  {fmt(rep,'direct')}")
        print(f"  FULL    {fmt(rep,'full')}   ==> {cls}")
    if args.runs > 1:
        fs = [r['full']['result']['speedup'] for r in reps if r['full']['result']]
        strat = set(r['full']['strategy'] for r in reps)
        print("-"*78)
        if fs: print(f"  FULL speedup range: {min(fs):.2f}x .. {max(fs):.2f}x")
        print(f"  FULL strategies seen: {strat}")
        # honest per-run classification (not just FULL<DIRECT)
        cats = [classify(r['full']['result']['speedup'] if r['full']['result'] else None,
                         r['direct']['result']['speedup'] if r['direct']['result'] else 1.0) for r in reps]
        print(f"  FULL per-run class: {cats}")
        print(f"  -> over-correction is {'STABLE' if all(c!='comparable' for c in cats) else ('INTERMITTENT' if any(c!='comparable' for c in cats) else 'ABSENT')}")

if __name__ == "__main__":
    main()
