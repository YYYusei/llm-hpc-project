"""
CUDA Generation Test for HPCG SYMGS - Version 2
Provide multi-coloring strategy explicitly
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

SYMGS_SOURCE = '''
// HPCG Symmetric Gauss-Seidel (SYMGS)
// Original sequential version with data dependencies

int ComputeSYMGS_ref(const SparseMatrix & A, const Vector & r, Vector & x) {
  const local_int_t nrow = A.localNumberOfRows;
  double ** matrixDiagonal = A.matrixDiagonal;
  const double * const rv = r.values;
  double * const xv = x.values;

  // Forward sweep (row i depends on rows 0..i-1)
  for (local_int_t i=0; i < nrow; i++) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0];
    double sum = rv[i];
    for (int j=0; j < currentNumberOfNonzeros; j++) {
      sum -= currentValues[j] * xv[currentColIndices[j]];
    }
    sum += xv[i] * currentDiagonal;
    xv[i] = sum / currentDiagonal;
  }

  // Backward sweep (row i depends on rows i+1..n-1)
  for (local_int_t i=nrow-1; i >= 0; i--) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0];
    double sum = rv[i];
    for (int j=0; j < currentNumberOfNonzeros; j++) {
      sum -= currentValues[j] * xv[currentColIndices[j]];
    }
    sum += xv[i] * currentDiagonal;
    xv[i] = sum / currentDiagonal;
  }
  return 0;
}
'''

OPTIMIZATION_PROMPT_V2 = '''
Convert the SYMGS code to CUDA using **Multi-Coloring** parallelization.

## Original CPU Code:
```cpp
{source}
```

## Multi-Coloring Strategy (USE THIS):
The matrix has been pre-colored so that rows of the same color have NO dependencies.
You will receive:
- num_colors: total number of colors (e.g., 8 for 3D stencil)
- row_colors[i]: color of row i (0 to num_colors-1)

## Algorithm:
```
For each color c from 0 to num_colors-1:
    Launch kernel: process ALL rows where row_colors[i] == c (in parallel)
    Synchronize
```

## Data Format:
- nrow: number of rows
- max_nnz: max non-zeros per row (27)
- nnz_per_row[i]: actual non-zeros in row i
- col_ind[i * max_nnz + j]: column index
- values[i * max_nnz + j]: matrix value
- diag[i]: diagonal value (pre-extracted)
- row_colors[i]: color of row i
- num_colors: total colors
- r[]: RHS vector
- x[]: solution vector (in/out)

## Requirements:
1. Write a kernel that processes rows of ONE color
2. Write a host function `{func_name}` that:
   - Loops through colors 0 to num_colors-1
   - For FORWARD sweep: colors in order 0,1,2,...
   - For BACKWARD sweep: colors in reverse order
   - Launches kernel for each color
3. Kernel signature should be:
```cuda
   __global__ void symgs_kernel(int nrow, int max_nnz, int target_color,
       const int* row_colors, const int* nnz_per_row, const int* col_ind,
       const double* values, const double* diag, const double* r, double* x)
```
4. Host function signature:
```cuda
   void {func_name}(int nrow, int max_nnz, int num_colors,
       const int* row_colors, const int* nnz_per_row, const int* col_ind,
       const double* values, const double* diag, const double* r, double* x)
```

Output ONLY the CUDA code (kernel + host function):
```cuda
// Your multi-colored SYMGS implementation
```
'''

BENCHMARK_TEMPLATE_V2 = '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) {{ cudaError_t e=call; if(e!=cudaSuccess){{printf("CUDA_ERROR: %s\\n",cudaGetErrorString(e));exit(1);}} }}

// ============ LLM Generated SYMGS (Multi-Colored) ============
{optimized_kernel}

// ============ CPU Reference ============
void symgs_cpu(int nrow, int max_nnz, const int* nnz_per_row,
               const int* col_ind, const double* values, const double* diag,
               const double* r, double* x) {{
    // Forward sweep
    for (int i = 0; i < nrow; i++) {{
        double sum = r[i];
        int row_nnz = nnz_per_row[i];
        for (int j = 0; j < row_nnz; j++) {{
            int idx = i * max_nnz + j;
            sum -= values[idx] * x[col_ind[idx]];
        }}
        sum += x[i] * diag[i];
        x[i] = sum / diag[i];
    }}
    // Backward sweep
    for (int i = nrow - 1; i >= 0; i--) {{
        double sum = r[i];
        int row_nnz = nnz_per_row[i];
        for (int j = 0; j < row_nnz; j++) {{
            int idx = i * max_nnz + j;
            sum -= values[idx] * x[col_ind[idx]];
        }}
        sum += x[i] * diag[i];
        x[i] = sum / diag[i];
    }}
}}

// Simple coloring for 3D stencil (based on (i/nx + i/ny + i/nz) % num_colors)
void compute_colors(int nrow, int* row_colors, int num_colors) {{
    int nx = 50, ny = 50, nz = nrow / (50 * 50);
    if (nz < 1) nz = 1;
    for (int i = 0; i < nrow; i++) {{
        int iz = i / (nx * ny);
        int iy = (i % (nx * ny)) / nx;
        int ix = i % nx;
        row_colors[i] = (ix + iy + iz) % num_colors;
    }}
}}

double check_correctness(double* x1, double* x2, int n) {{
    double maxerr = 0.0;
    for (int i = 0; i < n; i++) {{
        double err = fabs(x1[i] - x2[i]);
        if (err > maxerr) maxerr = err;
    }}
    return maxerr;
}}

int main() {{
    int nrow = 50000;
    int max_nnz = 27;
    int num_colors = 8;  // For 3D 27-point stencil
    
    // Allocate host memory
    int* h_nnz_per_row = (int*)malloc(nrow * sizeof(int));
    int* h_col_ind = (int*)malloc(nrow * max_nnz * sizeof(int));
    double* h_values = (double*)malloc(nrow * max_nnz * sizeof(double));
    double* h_diag = (double*)malloc(nrow * sizeof(double));
    double* h_r = (double*)malloc(nrow * sizeof(double));
    double* h_x_cpu = (double*)malloc(nrow * sizeof(double));
    double* h_x_gpu = (double*)malloc(nrow * sizeof(double));
    int* h_row_colors = (int*)malloc(nrow * sizeof(int));
    
    // Compute colors
    compute_colors(nrow, h_row_colors, num_colors);
    
    // Initialize matrix (3D 27-point stencil pattern)
    srand(12345);
    for (int i = 0; i < nrow; i++) {{
        h_nnz_per_row[i] = 27;
        h_diag[i] = 26.0;
        h_r[i] = (double)rand() / RAND_MAX;
        h_x_cpu[i] = 0.0;
        h_x_gpu[i] = 0.0;
        
        for (int j = 0; j < max_nnz; j++) {{
            int idx = i * max_nnz + j;
            if (j == 13) {{
                h_col_ind[idx] = i;
                h_values[idx] = 26.0;
            }} else {{
                int offset = (j < 13) ? (j - 13) : (j - 13);
                int col = i + offset * 100 + (rand() % 10 - 5);
                if (col < 0) col = 0;
                if (col >= nrow) col = nrow - 1;
                h_col_ind[idx] = col;
                h_values[idx] = -1.0;
            }}
        }}
    }}
    
    // CPU benchmark
    clock_t cpu_start = clock();
    for (int iter = 0; iter < 3; iter++) {{
        for (int i = 0; i < nrow; i++) h_x_cpu[i] = 0.0;
        symgs_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_diag, h_r, h_x_cpu);
    }}
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0 / 3.0;
    
    // Allocate device memory
    int *d_nnz_per_row, *d_col_ind, *d_row_colors;
    double *d_values, *d_diag, *d_r, *d_x;
    CHECK_CUDA(cudaMalloc(&d_nnz_per_row, nrow * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, nrow * max_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nrow * max_nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_diag, nrow * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r, nrow * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, nrow * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_row_colors, nrow * sizeof(int)));
    
    CHECK_CUDA(cudaMemcpy(d_nnz_per_row, h_nnz_per_row, nrow * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, h_col_ind, nrow * max_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nrow * max_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_diag, h_diag, nrow * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_r, h_r, nrow * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_row_colors, h_row_colors, nrow * sizeof(int), cudaMemcpyHostToDevice));
    
    // GPU benchmark
    CHECK_CUDA(cudaMemset(d_x, 0, nrow * sizeof(double)));
    {func_name}(nrow, max_nnz, num_colors, d_row_colors, d_nnz_per_row, d_col_ind, d_values, d_diag, d_r, d_x);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int iter = 0; iter < 5; iter++) {{
        CHECK_CUDA(cudaMemset(d_x, 0, nrow * sizeof(double)));
        {func_name}(nrow, max_nnz, num_colors, d_row_colors, d_nnz_per_row, d_col_ind, d_values, d_diag, d_r, d_x);
    }}
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= 5;
    
    CHECK_CUDA(cudaMemcpy(h_x_gpu, d_x, nrow * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Note: Multi-colored GS gives slightly different results than sequential GS
    // We compare against CPU sequential as reference
    for (int i = 0; i < nrow; i++) h_x_cpu[i] = 0.0;
    symgs_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_diag, h_r, h_x_cpu);
    
    double err = check_correctness(h_x_cpu, h_x_gpu, nrow);
    
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\\n",
           cpu_ms, gpu_ms, cpu_ms/gpu_ms, err);
    
    // Cleanup
    free(h_nnz_per_row); free(h_col_ind); free(h_values);
    free(h_diag); free(h_r); free(h_x_cpu); free(h_x_gpu); free(h_row_colors);
    cudaFree(d_nnz_per_row); cudaFree(d_col_ind); cudaFree(d_values);
    cudaFree(d_diag); cudaFree(d_r); cudaFree(d_x); cudaFree(d_row_colors);
    
    return 0;
}}
'''


class SYMGSCUDATesterV2:
    """Test CUDA SYMGS with multi-coloring hint"""
    
    def __init__(self, output_dir: str = "results/cuda_symgs_v2"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.clients = {
            "gpt-4o": LLMClient(model="gpt-4o"),
            "gpt-5.2": LLMClient(model="gpt-5.2")
        }
    
    def extract_cuda_code(self, response: str) -> str:
        patterns = [
            r'```cuda\n(.*?)```',
            r'```cpp\n(.*?)```',
            r'```c\n(.*?)```',
            r'```\n(.*?)```'
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        return response.strip()
    
    def generate_kernel(self, model: str, func_name: str) -> Dict[str, Any]:
        logger.info(f"[{model}] Generating SYMGS kernel (with multi-coloring hint)...")
        
        prompt = OPTIMIZATION_PROMPT_V2.format(
            source=SYMGS_SOURCE,
            func_name=func_name
        )
        
        response = self.clients[model].chat(
            prompt=prompt,
            system_prompt="You are a CUDA expert. Follow the multi-coloring strategy exactly as specified."
        )
        
        code = self.extract_cuda_code(response.content)
        
        return {
            "code": code,
            "full_response": response.content,
            "cost": response.cost,
            "time": response.elapsed_time,
            "tokens": response.total_tokens
        }
    
    def fix_kernel(self, buggy_code: str, error_msg: str, func_name: str) -> Dict[str, Any]:
        logger.info("[gpt-5.2] Fixing SYMGS kernel...")
        
        prompt = f'''
Fix this CUDA SYMGS kernel:
```cuda
{buggy_code}
```

Error:
{error_msg[:500]}

Requirements:
1. Host function signature MUST be:
   void {func_name}(int nrow, int max_nnz, int num_colors,
       const int* row_colors, const int* nnz_per_row, const int* col_ind,
       const double* values, const double* diag, const double* r, double* x)
2. Use multi-coloring: loop through colors, launch kernel for each color

Output ONLY fixed code:
```cuda
```
'''
        
        response = self.clients["gpt-5.2"].chat(
            prompt=prompt,
            system_prompt="Fix the CUDA code. Match the exact function signature."
        )
        
        return {
            "code": self.extract_cuda_code(response.content),
            "cost": response.cost,
            "time": response.elapsed_time,
        }
    
    def compile_and_test(self, kernel_code: str, func_name: str, test_name: str) -> Tuple[bool, str, Optional[Dict]]:
        logger.info(f"[{test_name}] Compiling and testing...")
        
        benchmark_code = BENCHMARK_TEMPLATE_V2.format(
            optimized_kernel=kernel_code,
            func_name=func_name
        )
        
        cu_filename = f"{test_name}_benchmark.cu"
        exe_filename = f"{test_name}_benchmark"
        cu_file = os.path.join(self.output_dir, cu_filename)
        
        with open(cu_file, 'w') as f:
            f.write(benchmark_code)
        
        win_path = os.path.abspath(self.output_dir)
        wsl_path = "/mnt/" + win_path[0].lower() + win_path[2:].replace("\\", "/")
        
        compile_cmd = f'wsl -d Ubuntu-24.04 bash -c "cd {wsl_path} && /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 {cu_filename} -o {exe_filename} 2>&1"'
        compile_result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        
        if compile_result.returncode != 0:
            error_msg = compile_result.stdout + compile_result.stderr
            logger.warning(f"[{test_name}] Compilation failed: {error_msg[:300]}")
            return False, f"COMPILE_ERROR: {error_msg}", None
        
        run_cmd = f'wsl -d Ubuntu-24.04 bash -c "cd {wsl_path} && ./{exe_filename} 2>&1"'
        try:
            run_result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return False, "RUNTIME_ERROR: Timeout", None
        
        output = run_result.stdout + run_result.stderr
        
        if "CUDA_ERROR" in output:
            return False, f"RUNTIME_ERROR: {output}", None
        
        match = re.search(r'BENCHMARK_RESULT:cpu_ms=([\d.]+),gpu_ms=([\d.]+),speedup=([\d.]+),error=([\d.e+-]+)', output)
        if match:
            results = {
                "cpu_ms": float(match.group(1)),
                "gpu_ms": float(match.group(2)),
                "speedup": float(match.group(3)),
                "error": float(match.group(4))
            }
            logger.info(f"[{test_name}] Success: speedup={results['speedup']:.2f}x, error={results['error']:.2e}")
            return True, "SUCCESS", results
        
        return False, f"PARSE_ERROR: {output}", None
    
    def test_model(self, model: str) -> Dict[str, Any]:
        func_name = f"symgs_gpu_{model.replace('.', '_').replace('-', '_')}"
        
        result = {
            "model": model,
            "generation": None,
            "first_test": None,
            "fix_attempt": None,
            "final_status": "UNKNOWN",
            "total_cost": 0
        }
        
        gen = self.generate_kernel(model, func_name)
        result["generation"] = gen
        result["total_cost"] += gen["cost"]
        
        with open(os.path.join(self.output_dir, f"{model.replace('.', '_')}_generated.cu"), 'w') as f:
            f.write(gen["code"])
        
        success, error_msg, benchmark = self.compile_and_test(gen["code"], func_name, f"{model.replace('.', '_')}_v1")
        result["first_test"] = {"success": success, "error": error_msg if not success else None, "benchmark": benchmark}
        
        if success:
            result["final_status"] = "SUCCESS_FIRST_TRY"
            result["final_test"] = benchmark
            return result
        
        # Fix
        fix = self.fix_kernel(gen["code"], error_msg, func_name)
        result["fix_attempt"] = fix
        result["total_cost"] += fix["cost"]
        
        with open(os.path.join(self.output_dir, f"{model.replace('.', '_')}_fixed.cu"), 'w') as f:
            f.write(fix["code"])
        
        success, error_msg, benchmark = self.compile_and_test(fix["code"], func_name, f"{model.replace('.', '_')}_v2")
        
        if success:
            result["final_status"] = "SUCCESS_AFTER_FIX"
            result["final_test"] = benchmark
        else:
            result["final_status"] = "FAILED"
            result["final_error"] = error_msg
        
        return result
    
    def run(self):
        logger.info("=" * 60)
        logger.info("HPCG SYMGS CUDA Test V2 (with Multi-Coloring)")
        logger.info("=" * 60)
        
        results = {"timestamp": datetime.now().isoformat(), "models": {}}
        
        for model in ["gpt-4o", "gpt-5.2"]:
            logger.info(f"\n{'='*60}\nTesting: {model}\n{'='*60}")
            results["models"][model] = self.test_model(model)
        
        with open(os.path.join(self.output_dir, "symgs_v2_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary
        print("\n" + "=" * 70)
        print("HPCG SYMGS CUDA TEST V2 (Multi-Coloring)")
        print("=" * 70)
        
        for model, data in results["models"].items():
            print(f"\n### {model} ###")
            print(f"  Status: {data['final_status']}")
            print(f"  Cost: ${data['total_cost']:.4f}")
            
            bench = data.get("final_test") or (data.get("first_test", {}).get("benchmark"))
            if bench:
                print(f"  CPU: {bench['cpu_ms']:.2f}ms, GPU: {bench['gpu_ms']:.2f}ms")
                print(f"  Speedup: {bench['speedup']:.2f}x")
                print(f"  Error: {bench['error']:.2e}")
        
        print("\n" + "=" * 70)
        total_cost = sum(d["total_cost"] for d in results["models"].values())
        print(f"Total Cost: ${total_cost:.4f}")
        print("=" * 70)


if __name__ == "__main__":
    SYMGSCUDATesterV2().run()