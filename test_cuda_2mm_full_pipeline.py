"""
2MM Phase-4 GPU Pipeline (Option A: strategy NOT hardcoded) + MULTI-RUN mode (--runs N)

DIRECT : source only -> usually standard tiled GEMM(s) -> correct & fast.
FULL   : cascade analysis drives generation. 2mm is a shared over-correction
         source at classification level. Watch if FULL repeatedly emits a
         fused / over-engineered kernel slower than DIRECT.

--runs N re-runs the whole pipeline N times (LLM gen re-sampled each time even at
temp=0). Each run under results/cuda_2mm_pipeline/runK/. Stability table at end.

Usage (WSL2, from project root, OPENAI_API_KEY + PYTHONUTF8=1):
    python3 test_cuda_2mm_full_pipeline.py --runs 3
"""
import sys, os, re, json, argparse, subprocess, logging
from datetime import datetime
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.insert(0, 'src')
from llm_client import LLMClient

NI = NJ = NK = NL = 1024
NVCC_ARCH = "sm_86"
CORRECTNESS_TOL = 1e-3

MM2_CODE = '''
/* 2mm: D = alpha*A*B*C + beta*D   (tmp = alpha*A*B ; D = tmp*C + beta*D) */
void kernel_2mm(int ni, int nj, int nk, int nl,
                double alpha, double beta,
                double tmp[ni][nj],
                double A[ni][nk], double B[nk][nj],
                double C[nj][nl], double D[ni][nl]) {
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++) {
      tmp[i][j] = 0.0;
      for (int k = 0; k < nk; ++k)
        tmp[i][j] += alpha * A[i][k] * B[k][j];
    }
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++) {
      D[i][j] *= beta;
      for (int k = 0; k < nj; ++k)
        D[i][j] += tmp[i][k] * C[k][j];
    }
}
'''

HOST_CONTRACT = '''
The generated code MUST expose exactly this host launcher (plus any kernels):

    void mm2_gpu(int ni, int nj, int nk, int nl,
                 double alpha, double beta,
                 double* A, double* B, double* C, double* D);

where (all DEVICE pointers, row-major, allocated/initialized by caller):
  - A is ni*nk, B is nk*nj, C is nj*nl, D is ni*nl. Index (r,c) of R x Cc as ptr[r*Cc + c].
  - mm2_gpu computes tmp = alpha*A*B (tmp is ni*nj), then D = tmp*C + beta*D in
    place on D, then returns. Allocate tmp/scratch as device memory inside and free it.
  - Do NOT copy to/from host inside mm2_gpu.
  - Output ONLY a single ```cuda code block.
'''

BENCHMARK_TEMPLATE = r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define CHECK_CUDA(call) {{ cudaError_t e=call; if(e!=cudaSuccess){{printf("CUDA_ERROR: %s\n",cudaGetErrorString(e));exit(1);}} }}
#define NI {NI}
#define NJ {NJ}
#define NK {NK}
#define NL {NL}
// ===== LLM Generated (kernels + host launcher mm2_gpu) =====
{generated}
// ===== Strict serial CPU reference =====
static void mm2_cpu(int ni,int nj,int nk,int nl,double alpha,double beta,
                    double* A,double* B,double* C,double* D,double* tmp) {{
    for (int i=0;i<ni;i++) for (int j=0;j<nj;j++) {{
        tmp[i*nj+j]=0.0;
        for (int k=0;k<nk;k++) tmp[i*nj+j]+=alpha*A[i*nk+k]*B[k*nj+j];
    }}
    for (int i=0;i<ni;i++) for (int j=0;j<nl;j++) {{
        D[i*nl+j]*=beta;
        for (int k=0;k<nj;k++) D[i*nl+j]+=tmp[i*nj+k]*C[k*nl+j];
    }}
}}
static void init_mats(int ni,int nj,int nk,int nl,double* A,double* B,double* C,double* D) {{
    for (int i=0;i<ni;i++) for (int k=0;k<nk;k++) A[i*nk+k]=(double)((i*k+1)%100)/ni;
    for (int k=0;k<nk;k++) for (int j=0;j<nj;j++) B[k*nj+j]=(double)((k+j)%100)/nj;
    for (int j=0;j<nj;j++) for (int l=0;l<nl;l++) C[j*nl+l]=(double)((j*l+2)%100)/nl;
    for (int i=0;i<ni;i++) for (int l=0;l<nl;l++) D[i*nl+l]=(double)((i+l)%100)/nl;
}}
static double max_abs_diff(const double* a,const double* b,int n){{double m=0;for(int k=0;k<n;k++){{double e=fabs(a[k]-b[k]);if(e>m)m=e;}}return m;}}
int main() {{
    int ni=NI,nj=NJ,nk=NK,nl=NL; double alpha=1.5, beta=1.2;
    double* A=(double*)malloc((size_t)ni*nk*sizeof(double));
    double* B=(double*)malloc((size_t)nk*nj*sizeof(double));
    double* C=(double*)malloc((size_t)nj*nl*sizeof(double));
    double* D0=(double*)malloc((size_t)ni*nl*sizeof(double));
    double* Dc=(double*)malloc((size_t)ni*nl*sizeof(double));
    double* Dg=(double*)malloc((size_t)ni*nl*sizeof(double));
    double* tmp=(double*)malloc((size_t)ni*nj*sizeof(double));
    init_mats(ni,nj,nk,nl,A,B,C,D0);
    for (size_t k=0;k<(size_t)ni*nl;k++) Dc[k]=D0[k];
    clock_t cs=clock(); mm2_cpu(ni,nj,nk,nl,alpha,beta,A,B,C,Dc,tmp); clock_t ce=clock();
    double cpu_ms=(double)(ce-cs)/CLOCKS_PER_SEC*1000.0;
    double *dA,*dB,*dC,*dD;
    CHECK_CUDA(cudaMalloc(&dA,(size_t)ni*nk*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dB,(size_t)nk*nj*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dC,(size_t)nj*nl*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dD,(size_t)ni*nl*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(dA,A,(size_t)ni*nk*sizeof(double),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,B,(size_t)nk*nj*sizeof(double),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC,C,(size_t)nj*nl*sizeof(double),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dD,D0,(size_t)ni*nl*sizeof(double),cudaMemcpyHostToDevice));
    mm2_gpu(ni,nj,nk,nl,alpha,beta,dA,dB,dC,dD);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Dg,dD,(size_t)ni*nl*sizeof(double),cudaMemcpyDeviceToHost));
    cudaEvent_t st,sp; cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    for (int r=0;r<5;r++) {{
        CHECK_CUDA(cudaMemcpy(dD,D0,(size_t)ni*nl*sizeof(double),cudaMemcpyHostToDevice));
        mm2_gpu(ni,nj,nk,nl,alpha,beta,dA,dB,dC,dD);
    }}
    cudaEventRecord(sp); cudaEventSynchronize(sp);
    float gpu_ms=0.0f; cudaEventElapsedTime(&gpu_ms,st,sp); gpu_ms/=5.0f;
    double err=max_abs_diff(Dc,Dg,ni*nl);
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\n",
           cpu_ms,gpu_ms,cpu_ms/(double)gpu_ms,err);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dD);
    free(A);free(B);free(C);free(D0);free(Dc);free(Dg);free(tmp);
    return 0;
}}
'''

def strategy_tag(code: str) -> str:
    c = code.lower(); tags = []
    if "fused" in c: tags.append("fused")
    if "__shared__" in c: tags.append("shared-mem")
    tags.append(f"{len(re.findall(r'__global__', code))}kernels")
    return "+".join(tags) if tags else "plain"

class MM2Pipeline:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.clients = {"gpt-4o": LLMClient(model="gpt-4o"), "gpt-5.2": LLMClient(model="gpt-5.2")}
        self.total_cost = 0.0
    def gen_direct(self) -> str:
        logger.info("[DIRECT] generating CUDA from source only...")
        p = f"\nTranslate the following C kernel into a CUDA GPU implementation.\n\n```c\n{MM2_CODE}\n```\n\n{HOST_CONTRACT}\n"
        r = self.clients["gpt-5.2"].chat(p, system_prompt="Generate clean, working CUDA code.")
        self.total_cost += r.cost; return self._extract(r.content)
    def stage1(self) -> str:
        logger.info("[FULL/Stage1] GPT-4o analyzing...")
        p = f'''
Analyze this HPC kernel for GPU parallelization:

```c
{MM2_CODE}
```

Identify:
1. Performance bottleneck type (compute-bound or memory-bound) and why.
2. Any data dependencies relevant to parallelization.
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
Review the following analysis of a chained dense matrix-multiply kernel (2mm) and
produce a CONCRETE GPU parallelization strategy. First validate whether the
previous bottleneck and dependency analysis is correct; if wrong or incomplete,
correct it. Then give a specific GPU scheme.

## Previous analysis (Stage 1):
{s1}

## Original kernel:
```c
{MM2_CODE}
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
Using ONLY the parallelization strategy below, implement the 2mm kernel in CUDA.
Follow the strategy's chosen approach faithfully; do not substitute a different scheme.

## Strategy (from cascade analysis):
{s2}

## Original kernel (for reference):
```c
{MM2_CODE}
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
        src = BENCHMARK_TEMPLATE.format(generated=generated, NI=NI, NJ=NJ, NK=NK, NL=NL)
        cu = os.path.join(self.output_dir, f"2mm_{tag}.cu")
        binp = os.path.join(self.output_dir, f"2mm_{tag}")
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
        rep: Dict[str, Any] = {"kernel": "2mm", "timestamp": datetime.now().isoformat()}
        dc = self.gen_direct()
        ok, msg, res = self.compile_and_run(dc, "direct")
        rep["direct"] = {"compiled_ran": ok, "msg": msg, "result": res, "strategy": strategy_tag(dc)}
        logger.info(f"[DIRECT] {msg} | {res} | {rep['direct']['strategy']}")
        s1 = self.stage1(); s2 = self.stage2(s1); fc = self.gen_full(s2)
        ok2, msg2, res2 = self.compile_and_run(fc, "full")
        rep["full"] = {"compiled_ran": ok2, "msg": msg2, "result": res2, "strategy": strategy_tag(fc), "stage1": s1, "stage2": s2}
        logger.info(f"[FULL] {msg2} | {res2} | {rep['full']['strategy']}")
        rep["total_cost"] = self.total_cost
        with open(os.path.join(self.output_dir, "2mm_pipeline_report.json"), "w", encoding="utf-8") as f:
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
    base = "results/cuda_2mm_pipeline"; reps: List[Dict[str, Any]] = []
    for k in range(1, args.runs + 1):
        outdir = base if args.runs == 1 else f"{base}/run{k}"
        logger.info(f"\n########## 2MM RUN {k}/{args.runs} -> {outdir} ##########")
        reps.append(MM2Pipeline(outdir).one_run())
    print("\n" + "="*78); print(f"2MM Phase-4 stability summary ({args.runs} run(s))"); print("="*78)
    for i, rep in enumerate(reps, 1):
        print(f"--- run {i} ---")
        print(f"  DIRECT  {fmt(rep,'direct')}")
        print(f"  FULL    {fmt(rep,'full')}")
    if args.runs > 1:
        fs = [r['full']['result']['speedup'] for r in reps if r['full']['result']]
        ds = [r['direct']['result']['speedup'] for r in reps if r['direct']['result']]
        strat = set(r['full']['strategy'] for r in reps)
        print("-"*78)
        if ds: print(f"  DIRECT speedup range: {min(ds):.2f}x .. {max(ds):.2f}x")
        if fs: print(f"  FULL   speedup range: {min(fs):.2f}x .. {max(fs):.2f}x")
        print(f"  FULL strategies seen: {strat}")
        if fs and ds:
            over = all(f < d for f, d in zip(fs, ds))
            print(f"  Over-correction reproducible (FULL < DIRECT every run): {over}")

if __name__ == "__main__":
    main()
