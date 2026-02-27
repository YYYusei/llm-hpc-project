"""
Full Pipeline Test for All Kernels
Compare: Direct Generation vs Full Pipeline (Cascaded Analysis → CUDA)
"""

import sys
import os
import json
import re
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, 'src')
from llm_client import LLMClient

# ============ Kernel Definitions ============

KERNELS = {
    "minimd": {
        "name": "miniMD LJ Force",
        "source": '''
// miniMD Lennard-Jones Force Calculation
void ForceLJ::compute(Atom &atom, Neighbor &neighbor) {
  int nlocal = atom.nlocal;
  int* neighs;
  MMD_float* x = atom.x;
  MMD_float* f = atom.f;
  int* type = atom.type;

  for(int i = 0; i < nlocal; i++) {
    neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    int numneigh = neighbor.numneigh[i];
    MMD_float xtmp = x[i * PAD + 0];
    MMD_float ytmp = x[i * PAD + 1];
    MMD_float ztmp = x[i * PAD + 2];
    int type_i = type[i];
    MMD_float fix = 0.0, fiy = 0.0, fiz = 0.0;

    for(int k = 0; k < numneigh; k++) {
      int j = neighs[k];
      MMD_float delx = xtmp - x[j * PAD + 0];
      MMD_float dely = ytmp - x[j * PAD + 1];
      MMD_float delz = ztmp - x[j * PAD + 2];
      int type_j = type[j];
      MMD_float rsq = delx*delx + dely*dely + delz*delz;
      int type_ij = type_i * ntypes + type_j;

      if(rsq < cutforcesq[type_ij]) {
        MMD_float sr2 = 1.0 / rsq;
        MMD_float sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
        MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;
      }
    }
    f[i * PAD + 0] = fix;
    f[i * PAD + 1] = fiy;
    f[i * PAD + 2] = fiz;
  }
}
''',
        "func_name": "lj_force_pipeline",
        "benchmark_template": '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) {{ cudaError_t e=call; if(e!=cudaSuccess){{printf("CUDA_ERROR: %s\\n",cudaGetErrorString(e));exit(1);}} }}

{optimized_kernel}

void lj_force_cpu(int nlocal, int ntypes, const double* x, double* f, const int* type,
    const int* neighbors, const int* numneigh, int maxneighs,
    const double* cutforcesq, const double* epsilon, const double* sigma6) {{
    for (int i = 0; i < nlocal; i++) {{
        double xtmp = x[i*4+0], ytmp = x[i*4+1], ztmp = x[i*4+2];
        int type_i = type[i];
        double fix = 0, fiy = 0, fiz = 0;
        for (int k = 0; k < numneigh[i]; k++) {{
            int j = neighbors[i * maxneighs + k];
            double delx = xtmp - x[j*4+0];
            double dely = ytmp - x[j*4+1];
            double delz = ztmp - x[j*4+2];
            double rsq = delx*delx + dely*dely + delz*delz;
            int type_ij = type_i * ntypes + type[j];
            if (rsq < cutforcesq[type_ij]) {{
                double sr2 = 1.0/rsq;
                double sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
                double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
                fix += delx * force;
                fiy += dely * force;
                fiz += delz * force;
            }}
        }}
        f[i*4+0] = fix; f[i*4+1] = fiy; f[i*4+2] = fiz; f[i*4+3] = 0;
    }}
}}

int main() {{
    int nlocal = 100000, ntypes = 1, maxneighs = 128;
    double *h_x, *h_f_cpu, *h_f_gpu, *h_cutforcesq, *h_epsilon, *h_sigma6;
    int *h_type, *h_neighbors, *h_numneigh;
    h_x = (double*)malloc(nlocal * 4 * sizeof(double));
    h_f_cpu = (double*)malloc(nlocal * 4 * sizeof(double));
    h_f_gpu = (double*)malloc(nlocal * 4 * sizeof(double));
    h_type = (int*)malloc(nlocal * sizeof(int));
    h_neighbors = (int*)malloc(nlocal * maxneighs * sizeof(int));
    h_numneigh = (int*)malloc(nlocal * sizeof(int));
    h_cutforcesq = (double*)malloc(ntypes * ntypes * sizeof(double));
    h_epsilon = (double*)malloc(ntypes * ntypes * sizeof(double));
    h_sigma6 = (double*)malloc(ntypes * ntypes * sizeof(double));
    
    srand(12345);
    for (int i = 0; i < nlocal; i++) {{
        h_x[i*4+0] = (double)rand()/RAND_MAX * 100;
        h_x[i*4+1] = (double)rand()/RAND_MAX * 100;
        h_x[i*4+2] = (double)rand()/RAND_MAX * 100;
        h_x[i*4+3] = 0;
        h_type[i] = 0;
        h_numneigh[i] = 50 + rand() % 20;
        for (int k = 0; k < h_numneigh[i]; k++)
            h_neighbors[i * maxneighs + k] = rand() % nlocal;
    }}
    h_cutforcesq[0] = 16.0; h_epsilon[0] = 1.0; h_sigma6[0] = 1.0;
    
    clock_t cpu_start = clock();
    for (int iter = 0; iter < 3; iter++) lj_force_cpu(nlocal, ntypes, h_x, h_f_cpu, h_type, h_neighbors, h_numneigh, maxneighs, h_cutforcesq, h_epsilon, h_sigma6);
    double cpu_ms = (double)(clock() - cpu_start) / CLOCKS_PER_SEC * 1000.0 / 3.0;
    
    double *d_x, *d_f, *d_cutforcesq, *d_epsilon, *d_sigma6;
    int *d_type, *d_neighbors, *d_numneigh;
    CHECK_CUDA(cudaMalloc(&d_x, nlocal * 4 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_f, nlocal * 4 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_type, nlocal * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_neighbors, nlocal * maxneighs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_numneigh, nlocal * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cutforcesq, ntypes * ntypes * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_epsilon, ntypes * ntypes * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_sigma6, ntypes * ntypes * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, nlocal * 4 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_type, h_type, nlocal * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_neighbors, h_neighbors, nlocal * maxneighs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_numneigh, h_numneigh, nlocal * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cutforcesq, h_cutforcesq, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_epsilon, h_epsilon, sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sigma6, h_sigma6, sizeof(double), cudaMemcpyHostToDevice));
    
    int bs = 256, nb = (nlocal + bs - 1) / bs;
    {func_name}<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neighbors, d_numneigh, maxneighs, d_cutforcesq, d_epsilon, d_sigma6);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int iter = 0; iter < 10; iter++)
        {func_name}<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neighbors, d_numneigh, maxneighs, d_cutforcesq, d_epsilon, d_sigma6);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, start, stop); gpu_ms /= 10;
    
    CHECK_CUDA(cudaMemcpy(h_f_gpu, d_f, nlocal * 4 * sizeof(double), cudaMemcpyDeviceToHost));
    double maxerr = 0;
    for (int i = 0; i < nlocal * 4; i++) {{ double e = fabs(h_f_cpu[i] - h_f_gpu[i]); if (e > maxerr) maxerr = e; }}
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\\n", cpu_ms, gpu_ms, cpu_ms/gpu_ms, maxerr);
    return 0;
}}
''',
        "kernel_signature": '''
__global__ void {func_name}(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6)
'''
    },
    
    "spmv": {
        "name": "HPCG SPMV",
        "source": '''
// HPCG Sparse Matrix-Vector Multiplication
int ComputeSPMV_ref(const SparseMatrix & A, Vector & x, Vector & y) {
  const local_int_t nrow = A.localNumberOfRows;
  double * const yv = y.values;
  double * const xv = x.values;

  for (local_int_t i=0; i< nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }
  return 0;
}
''',
        "func_name": "spmv_pipeline",
        "benchmark_template": '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) {{ cudaError_t e=call; if(e!=cudaSuccess){{printf("CUDA_ERROR: %s\\n",cudaGetErrorString(e));exit(1);}} }}

{optimized_kernel}

void spmv_cpu(int nrow, int max_nnz, const int* nnz_per_row, const int* col_ind, const double* values, const double* x, double* y) {{
    for (int i = 0; i < nrow; i++) {{
        double sum = 0.0;
        for (int j = 0; j < nnz_per_row[i]; j++) {{
            int idx = i * max_nnz + j;
            sum += values[idx] * x[col_ind[idx]];
        }}
        y[i] = sum;
    }}
}}

int main() {{
    int nrow = 100000, max_nnz = 27;
    int *h_nnz_per_row = (int*)malloc(nrow * sizeof(int));
    int *h_col_ind = (int*)malloc(nrow * max_nnz * sizeof(int));
    double *h_values = (double*)malloc(nrow * max_nnz * sizeof(double));
    double *h_x = (double*)malloc(nrow * sizeof(double));
    double *h_y_cpu = (double*)malloc(nrow * sizeof(double));
    double *h_y_gpu = (double*)malloc(nrow * sizeof(double));
    
    srand(12345);
    for (int i = 0; i < nrow; i++) {{
        h_nnz_per_row[i] = 27;
        h_x[i] = (double)rand() / RAND_MAX;
        for (int j = 0; j < max_nnz; j++) {{
            int idx = i * max_nnz + j;
            h_col_ind[idx] = (i + j - 13 + nrow) % nrow;
            h_values[idx] = (j == 13) ? 26.0 : -1.0;
        }}
    }}
    
    clock_t cpu_start = clock();
    for (int iter = 0; iter < 10; iter++) spmv_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_x, h_y_cpu);
    double cpu_ms = (double)(clock() - cpu_start) / CLOCKS_PER_SEC * 1000.0 / 10.0;
    
    int *d_nnz_per_row, *d_col_ind;
    double *d_values, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_nnz_per_row, nrow * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, nrow * max_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nrow * max_nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, nrow * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, nrow * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_nnz_per_row, h_nnz_per_row, nrow * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, h_col_ind, nrow * max_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nrow * max_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, nrow * sizeof(double), cudaMemcpyHostToDevice));
    
    int bs = 256, nb = (nrow + bs - 1) / bs;
    {func_name}<<<nb, bs>>>(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int iter = 0; iter < 20; iter++)
        {func_name}<<<nb, bs>>>(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, start, stop); gpu_ms /= 20;
    
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, nrow * sizeof(double), cudaMemcpyDeviceToHost));
    double maxerr = 0;
    for (int i = 0; i < nrow; i++) {{ double e = fabs(h_y_cpu[i] - h_y_gpu[i]); if (e > maxerr) maxerr = e; }}
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\\n", cpu_ms, gpu_ms, cpu_ms/gpu_ms, maxerr);
    return 0;
}}
''',
        "kernel_signature": '''
__global__ void {func_name}(int nrow, int max_nnz,
    const int* __restrict__ nnz_per_row, const int* __restrict__ col_ind,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
'''
    }
}


class FullPipelineTester:
    """Test full pipeline for all kernels"""
    
    def __init__(self, output_dir: str = "results/full_pipeline_comparison"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.clients = {
            "gpt-4o": LLMClient(model="gpt-4o"),
            "gpt-5.2": LLMClient(model="gpt-5.2")
        }
    
    def extract_cuda_code(self, response: str) -> str:
        for pattern in [r'```cuda\n(.*?)```', r'```cpp\n(.*?)```', r'```\n(.*?)```']:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        return response.strip()
    
    def stage1_analysis(self, kernel_name: str, source: str) -> Dict[str, Any]:
        """Stage 1: GPT-4o analyzes the code"""
        logger.info(f"[{kernel_name}] Stage 1: GPT-4o analyzing...")
        
        prompt = f'''
Analyze this HPC kernel for GPU parallelization:
```cpp
{source}
```

Identify:
1. Hotspots and bottleneck type (compute/memory bound)
2. Data dependencies
3. Parallelization potential
4. Recommended GPU strategy

Output JSON format.
'''
        response = self.clients["gpt-4o"].chat(prompt, system_prompt="You are an HPC performance expert.")
        
        return {
            "model": "gpt-4o",
            "response": response.content,
            "cost": response.cost,
            "time": response.elapsed_time
        }
    
    def stage2_strategy(self, kernel_name: str, source: str, stage1: str) -> Dict[str, Any]:
        """Stage 2: GPT-5.2 provides detailed strategy"""
        logger.info(f"[{kernel_name}] Stage 2: GPT-5.2 deep analysis...")
        
        prompt = f'''
Review and enhance this analysis, then provide DETAILED CUDA implementation strategy:

## Previous Analysis:
{stage1}

## Original Code:
```cpp
{source}
```

Provide:
1. Validation of bottleneck analysis
2. Specific CUDA optimization techniques to use
3. Memory access patterns to optimize
4. Thread/block organization

Output JSON with detailed strategy.
'''
        response = self.clients["gpt-5.2"].chat(prompt, system_prompt="You are a GPU optimization expert.")
        
        return {
            "model": "gpt-5.2",
            "response": response.content,
            "cost": response.cost,
            "time": response.elapsed_time
        }
    
    def stage3_generate(self, kernel_name: str, source: str, strategy: str, 
                        func_name: str, signature: str) -> Dict[str, Any]:
        """Stage 3: Generate CUDA code based on strategy"""
        logger.info(f"[{kernel_name}] Stage 3: Generating CUDA...")
        
        prompt = f'''
Based on this optimization strategy, generate an optimized CUDA kernel:

## Strategy:
{strategy}

## Original Code:
```cpp
{source}
```

## Requirements:
1. Function signature MUST be exactly:
{signature}

2. Apply the optimization techniques from the strategy
3. Use __ldg, fma, vectorized loads where appropriate

Output ONLY the CUDA kernel code:
```cuda
```
'''
        response = self.clients["gpt-5.2"].chat(prompt, system_prompt="Generate optimized CUDA code.")
        code = self.extract_cuda_code(response.content)
        
        return {
            "model": "gpt-5.2",
            "code": code,
            "cost": response.cost,
            "time": response.elapsed_time
        }
    
    def compile_and_test(self, kernel_name: str, code: str, func_name: str, 
                         benchmark_template: str, test_name: str) -> Tuple[bool, str, Optional[Dict]]:
        """Compile and benchmark"""
        logger.info(f"[{kernel_name}] Compiling {test_name}...")
        
        benchmark_code = benchmark_template.format(optimized_kernel=code, func_name=func_name)
        
        kernel_dir = os.path.join(self.output_dir, kernel_name)
        os.makedirs(kernel_dir, exist_ok=True)
        
        cu_file = os.path.join(kernel_dir, f"{test_name}.cu")
        with open(cu_file, 'w') as f:
            f.write(benchmark_code)
        
        win_path = os.path.abspath(kernel_dir)
        wsl_path = "/mnt/" + win_path[0].lower() + win_path[2:].replace("\\", "/")
        
        compile_cmd = f'wsl -d Ubuntu-24.04 bash -c "cd {wsl_path} && /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 {test_name}.cu -o {test_name} 2>&1"'
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            error = result.stdout + result.stderr
            logger.warning(f"[{kernel_name}] Compile failed: {error[:200]}")
            return False, error, None
        
        run_cmd = f'wsl -d Ubuntu-24.04 bash -c "cd {wsl_path} && ./{test_name} 2>&1"'
        try:
            result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False, "Timeout", None
        
        output = result.stdout + result.stderr
        if "CUDA_ERROR" in output:
            return False, output, None
        
        match = re.search(r'BENCHMARK_RESULT:cpu_ms=([\d.]+),gpu_ms=([\d.]+),speedup=([\d.]+),error=([\d.e+-]+)', output)
        if match:
            return True, "SUCCESS", {
                "cpu_ms": float(match.group(1)),
                "gpu_ms": float(match.group(2)),
                "speedup": float(match.group(3)),
                "error": float(match.group(4))
            }
        return False, output, None
    
    def fix_code(self, code: str, error: str, func_name: str, signature: str) -> Tuple[str, float]:
        """Fix compilation errors"""
        logger.info("[Fix] GPT-5.2 fixing...")
        
        prompt = f'''
Fix this CUDA kernel:
```cuda
{code}
```

Error: {error[:400]}

Function signature MUST be exactly:
{signature}

Output ONLY fixed code:
```cuda
```
'''
        response = self.clients["gpt-5.2"].chat(prompt)
        return self.extract_cuda_code(response.content), response.cost
    
    def test_kernel(self, kernel_id: str) -> Dict[str, Any]:
        """Run full pipeline for one kernel"""
        kernel = KERNELS[kernel_id]
        func_name = kernel["func_name"]
        signature = kernel["kernel_signature"].format(func_name=func_name)
        
        result = {
            "kernel": kernel["name"],
            "stages": {},
            "total_cost": 0,
            "final_status": "UNKNOWN"
        }
        
        # Stage 1
        s1 = self.stage1_analysis(kernel_id, kernel["source"])
        result["stages"]["stage1"] = s1
        result["total_cost"] += s1["cost"]
        
        # Stage 2
        s2 = self.stage2_strategy(kernel_id, kernel["source"], s1["response"])
        result["stages"]["stage2"] = s2
        result["total_cost"] += s2["cost"]
        
        # Stage 3
        s3 = self.stage3_generate(kernel_id, kernel["source"], s2["response"], func_name, signature)
        result["stages"]["stage3"] = s3
        result["total_cost"] += s3["cost"]
        
        # Save generated code
        kernel_dir = os.path.join(self.output_dir, kernel_id)
        os.makedirs(kernel_dir, exist_ok=True)
        with open(os.path.join(kernel_dir, "generated.cu"), 'w') as f:
            f.write(s3["code"])
        
        # Test
        success, error, benchmark = self.compile_and_test(
            kernel_id, s3["code"], func_name, kernel["benchmark_template"], "pipeline_v1"
        )
        
        if not success:
            fixed_code, fix_cost = self.fix_code(s3["code"], error, func_name, signature)
            result["total_cost"] += fix_cost
            with open(os.path.join(kernel_dir, "fixed.cu"), 'w') as f:
                f.write(fixed_code)
            success, error, benchmark = self.compile_and_test(
                kernel_id, fixed_code, func_name, kernel["benchmark_template"], "pipeline_v2"
            )
        
        if success:
            result["final_status"] = "SUCCESS"
            result["benchmark"] = benchmark
        else:
            result["final_status"] = "FAILED"
            result["error"] = error[:200]
        
        return result
    
    def run(self):
        """Run full pipeline for all kernels"""
        logger.info("=" * 60)
        logger.info("Full Pipeline Test: miniMD + SPMV")
        logger.info("=" * 60)
        
        results = {"timestamp": datetime.now().isoformat(), "kernels": {}}
        
        for kernel_id in ["minimd", "spmv"]:
            logger.info(f"\n{'='*60}\nTesting: {KERNELS[kernel_id]['name']}\n{'='*60}")
            results["kernels"][kernel_id] = self.test_kernel(kernel_id)
        
        # Save results
        with open(os.path.join(self.output_dir, "pipeline_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary
        print("\n" + "=" * 70)
        print("FULL PIPELINE RESULTS (Cascaded Analysis → CUDA)")
        print("=" * 70)
        
        for kernel_id, data in results["kernels"].items():
            print(f"\n### {data['kernel']} ###")
            print(f"  Status: {data['final_status']}")
            print(f"  Total Cost: ${data['total_cost']:.4f}")
            if data.get("benchmark"):
                b = data["benchmark"]
                print(f"  CPU: {b['cpu_ms']:.2f}ms, GPU: {b['gpu_ms']:.2f}ms")
                print(f"  Speedup: {b['speedup']:.2f}x")
                print(f"  Error: {b['error']:.2e}")
        
        # Comparison with direct generation
        print("\n" + "=" * 70)
        print("COMPARISON: Direct Generation vs Full Pipeline")
        print("=" * 70)
        print(f"{'Kernel':<20} {'Direct':>12} {'Pipeline':>12} {'Diff':>10}")
        print("-" * 54)
        
        direct_results = {"minimd": 14.34, "spmv": 10.30}
        for kernel_id, data in results["kernels"].items():
            direct = direct_results[kernel_id]
            pipeline = data.get("benchmark", {}).get("speedup", 0)
            diff = ((pipeline - direct) / direct * 100) if direct > 0 else 0
            print(f"{KERNELS[kernel_id]['name']:<20} {direct:>10.2f}x {pipeline:>10.2f}x {diff:>+9.1f}%")
        
        print("=" * 70)
        total_cost = sum(d["total_cost"] for d in results["kernels"].values())
        print(f"Total Cost: ${total_cost:.4f}")
        print("=" * 70)
        
        return results


if __name__ == "__main__":
    FullPipelineTester().run()