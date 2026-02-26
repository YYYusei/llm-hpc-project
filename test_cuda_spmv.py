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

# SPMV source code
SPMV_SOURCE = '''
// HPCG Sparse Matrix-Vector Multiplication (SPMV)
// Computes y = A * x where A is a sparse matrix in CSR-like format

// Data structures:
// - A.matrixValues[i]: array of non-zero values in row i
// - A.mtxIndL[i]: array of column indices for row i  
// - A.nonzerosInRow[i]: number of non-zeros in row i
// - x.values: input vector
// - y.values: output vector

int ComputeSPMV_ref(const SparseMatrix & A, Vector & x, Vector & y) {
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
  
  #pragma omp parallel for
  for (local_int_t i=0; i < nrow; i++) {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];
    
    for (int j=0; j < cur_nnz; j++)
      sum += cur_vals[j] * xv[cur_inds[j]];
    
    yv[i] = sum;
  }
  return 0;
}
'''

OPTIMIZATION_PROMPT = '''
Please convert the following HPCG SPMV (Sparse Matrix-Vector Multiplication) code to an optimized CUDA kernel.

## Original CPU Code:
```cpp
{source}
```

## Data Format (CSR-like):
- nrow: number of rows
- nnz_per_row[i]: number of non-zeros in row i (typically 27 for 3D stencil)
- col_ind[i * max_nnz + j]: column index of j-th non-zero in row i
- values[i * max_nnz + j]: value of j-th non-zero in row i
- x[]: input vector
- y[]: output vector (y = A * x)

## Info:
- GPU: RTX 3060 (SM 8.6)
- Typical problem: 100K - 1M rows
- Non-zeros per row: ~27 (3D 27-point stencil)
- Matrix is very sparse, memory-bound kernel

## Apply optimizations:
1. One thread per row (basic parallelization)
2. Use __ldg() for read-only data
3. Consider warp-level reduction if beneficial
4. Memory coalescing where possible

## Requirements:
1. Maintain numerical correctness
2. Function name must be: {func_name}
3. Output ONLY the kernel code
```cuda
// Your optimized SPMV kernel here
```
'''

FIX_PROMPT = '''
The following CUDA SPMV kernel has a bug.

## Buggy Code:
```cuda
{buggy_code}
```

## Error:
{error_msg}

IMPORTANT:
- Function name must be: {func_name}

Fix the bug and output ONLY the kernel code.
```cuda
// Fixed kernel here
```
'''

BENCHMARK_TEMPLATE = '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) {{ cudaError_t e=call; if(e!=cudaSuccess){{printf("CUDA_ERROR: %s\\n",cudaGetErrorString(e));exit(1);}} }}

// ============ Optimized SPMV Kernel ============
{optimized_kernel}

// ============ CPU Reference ============
void spmv_cpu(int nrow, int max_nnz, const int* nnz_per_row, 
              const int* col_ind, const double* values,
              const double* x, double* y) {{
    for (int i = 0; i < nrow; i++) {{
        double sum = 0.0;
        int row_nnz = nnz_per_row[i];
        for (int j = 0; j < row_nnz; j++) {{
            int idx = i * max_nnz + j;
            sum += values[idx] * x[col_ind[idx]];
        }}
        y[i] = sum;
    }}
}}

double check_correctness(double* y1, double* y2, int n) {{
    double maxerr = 0.0;
    for (int i = 0; i < n; i++) {{
        double err = fabs(y1[i] - y2[i]);
        if (err > maxerr) maxerr = err;
    }}
    return maxerr;
}}

int main() {{
    // Problem size: 100K rows, ~27 non-zeros per row (3D stencil)
    int nrow = 100000;
    int max_nnz = 27;
    int ncol = nrow;  // Square matrix
    
    // Allocate host memory
    int* h_nnz_per_row = (int*)malloc(nrow * sizeof(int));
    int* h_col_ind = (int*)malloc(nrow * max_nnz * sizeof(int));
    double* h_values = (double*)malloc(nrow * max_nnz * sizeof(double));
    double* h_x = (double*)malloc(ncol * sizeof(double));
    double* h_y_cpu = (double*)malloc(nrow * sizeof(double));
    double* h_y_gpu = (double*)malloc(nrow * sizeof(double));
    
    // Initialize data (simulate 3D 27-point stencil)
    srand(12345);
    for (int i = 0; i < nrow; i++) {{
        h_nnz_per_row[i] = 27;  // Fixed for stencil
        for (int j = 0; j < max_nnz; j++) {{
            int idx = i * max_nnz + j;
            // Diagonal-dominant pattern
            if (j == 13) {{  // Center point
                h_col_ind[idx] = i;
                h_values[idx] = 26.0;
            }} else {{
                // Random neighbor within bounds
                int offset = (j < 13) ? (j - 13) : (j - 13);
                int col = i + offset * 100 + (rand() % 10 - 5);
                if (col < 0) col = 0;
                if (col >= ncol) col = ncol - 1;
                h_col_ind[idx] = col;
                h_values[idx] = -1.0;
            }}
        }}
    }}
    
    for (int i = 0; i < ncol; i++) {{
        h_x[i] = (double)rand() / RAND_MAX;
    }}
    
    // CPU benchmark
    clock_t cpu_start = clock();
    for (int r = 0; r < 5; r++)
        spmv_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_x, h_y_cpu);
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0 / 5.0;
    
    // Allocate device memory
    int *d_nnz_per_row, *d_col_ind;
    double *d_values, *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_nnz_per_row, nrow * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_ind, nrow * max_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nrow * max_nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, ncol * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, nrow * sizeof(double)));
    
    CHECK_CUDA(cudaMemcpy(d_nnz_per_row, h_nnz_per_row, nrow * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, h_col_ind, nrow * max_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values, nrow * max_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, ncol * sizeof(double), cudaMemcpyHostToDevice));
    
    int bs = 256, nb = (nrow + bs - 1) / bs;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // GPU benchmark
    {func_name}<<<nb, bs>>>(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_x, d_y);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    cudaEventRecord(start);
    for (int r = 0; r < 10; r++)
        {func_name}<<<nb, bs>>>(nrow, max_nnz, d_nnz_per_row, d_col_ind, d_values, d_x, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    gpu_ms /= 10;
    
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y, nrow * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Check correctness
    double err = check_correctness(h_y_cpu, h_y_gpu, nrow);
    
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\\n",
           cpu_ms, gpu_ms, cpu_ms/gpu_ms, err);
    
    // Cleanup
    free(h_nnz_per_row); free(h_col_ind); free(h_values);
    free(h_x); free(h_y_cpu); free(h_y_gpu);
    cudaFree(d_nnz_per_row); cudaFree(d_col_ind); cudaFree(d_values);
    cudaFree(d_x); cudaFree(d_y);
    
    return 0;
}}
'''


class SPMVCUDATester:
    """Test CUDA SPMV kernel generation"""
    
    def __init__(self, output_dir: str = "results/cuda_spmv"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.clients = {
            "gpt-4o": LLMClient(model="gpt-4o"),
            "gpt-5.2": LLMClient(model="gpt-5.2")
        }
    
    def extract_cuda_code(self, response: str) -> str:
        """Extract CUDA code from LLM response"""
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
        """Generate optimized SPMV kernel"""
        logger.info(f"[{model}] Generating SPMV kernel...")
        
        prompt = OPTIMIZATION_PROMPT.format(
            source=SPMV_SOURCE,
            func_name=func_name
        )
        
        response = self.clients[model].chat(
            prompt=prompt,
            system_prompt="You are a CUDA optimization expert specializing in sparse matrix computations."
        )
        
        code = self.extract_cuda_code(response.content)
        
        return {
            "code": code,
            "cost": response.cost,
            "time": response.elapsed_time,
            "tokens": response.total_tokens
        }
    
    def fix_kernel(self, buggy_code: str, error_msg: str, func_name: str) -> Dict[str, Any]:
        """Let GPT-5.2 fix buggy kernel"""
        logger.info("[gpt-5.2] Fixing SPMV kernel...")
        
        prompt = FIX_PROMPT.format(
            buggy_code=buggy_code,
            error_msg=error_msg[:500],
            func_name=func_name
        )
        
        response = self.clients["gpt-5.2"].chat(
            prompt=prompt,
            system_prompt="You are a CUDA debugging expert. Fix the bug and provide working code."
        )
        
        code = self.extract_cuda_code(response.content)
        
        return {
            "code": code,
            "cost": response.cost,
            "time": response.elapsed_time,
            "tokens": response.total_tokens
        }
    
    def compile_and_test(self, kernel_code: str, func_name: str, test_name: str) -> Tuple[bool, str, Optional[Dict]]:
        """Compile and benchmark the kernel"""
        logger.info(f"[{test_name}] Compiling and testing...")
        
        benchmark_code = BENCHMARK_TEMPLATE.format(
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
            logger.warning(f"[{test_name}] Compilation failed: {error_msg[:200]}")
            return False, f"COMPILE_ERROR: {error_msg}", None
        
        run_cmd = f'wsl -d Ubuntu-24.04 bash -c "cd {wsl_path} && ./{exe_filename} 2>&1"'
        try:
            run_result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False, "RUNTIME_ERROR: Timeout", None
        
        output = run_result.stdout + run_result.stderr
        
        if "CUDA_ERROR" in output:
            logger.warning(f"[{test_name}] Runtime error: {output[:200]}")
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
        """Full test for one model"""
        func_name = f"spmv_kernel_{model.replace('.', '_').replace('-', '_')}"
        
        result = {
            "model": model,
            "func_name": func_name,
            "generation": None,
            "first_test": None,
            "fix_attempt": None,
            "final_test": None,
            "final_status": "UNKNOWN",
            "total_cost": 0
        }
        
        # Generate
        gen = self.generate_kernel(model, func_name)
        result["generation"] = gen
        result["total_cost"] += gen["cost"]
        
        with open(os.path.join(self.output_dir, f"{model.replace('.', '_')}_generated.cu"), 'w') as f:
            f.write(gen["code"])
        
        # First test
        success, error_msg, benchmark = self.compile_and_test(
            gen["code"], func_name, f"{model.replace('.', '_')}_v1"
        )
        result["first_test"] = {
            "success": success,
            "error": error_msg if not success else None,
            "benchmark": benchmark
        }
        
        if success:
            result["final_status"] = "SUCCESS_FIRST_TRY"
            result["final_test"] = benchmark
            return result
        
        # Fix with GPT-5.2
        logger.info(f"[{model}] First attempt failed, asking gpt-5.2 to fix...")
        fix = self.fix_kernel(gen["code"], error_msg, func_name)
        result["fix_attempt"] = {"fixer": "gpt-5.2", **fix}
        result["total_cost"] += fix["cost"]
        
        with open(os.path.join(self.output_dir, f"{model.replace('.', '_')}_fixed.cu"), 'w') as f:
            f.write(fix["code"])
        
        # Test fixed
        success, error_msg, benchmark = self.compile_and_test(
            fix["code"], func_name, f"{model.replace('.', '_')}_v2"
        )
        
        if success:
            result["final_status"] = "SUCCESS_AFTER_FIX"
            result["final_test"] = benchmark
        else:
            result["final_status"] = "FAILED"
            result["final_error"] = error_msg
        
        return result
    
    def run(self):
        """Run full test"""
        logger.info("=" * 60)
        logger.info("HPCG SPMV CUDA Generation Test")
        logger.info("=" * 60)
        
        results = {"timestamp": datetime.now().isoformat(), "models": {}}
        
        for model in ["gpt-4o", "gpt-5.2"]:
            logger.info(f"\n{'='*60}\nTesting: {model}\n{'='*60}")
            results["models"][model] = self.test_model(model)
        
        # Save results
        with open(os.path.join(self.output_dir, "spmv_test_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary
        print("\n" + "=" * 70)
        print("HPCG SPMV CUDA TEST SUMMARY")
        print("=" * 70)
        
        for model, data in results["models"].items():
            print(f"\n### {model} ###")
            print(f"  Status: {data['final_status']}")
            print(f"  Cost: ${data['total_cost']:.4f}")
            
            bench = data.get("final_test") or data.get("first_test", {}).get("benchmark")
            if bench:
                print(f"  CPU: {bench['cpu_ms']:.2f}ms, GPU: {bench['gpu_ms']:.2f}ms")
                print(f"  Speedup: {bench['speedup']:.2f}x")
                print(f"  Error: {bench['error']:.2e}")
        
        print("\n" + "=" * 70)
        total_cost = sum(d.get("total_cost", 0) for d in results["models"].values())
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Results saved to: {self.output_dir}/spmv_test_results.json")
        print("=" * 70)
        
        return results


if __name__ == "__main__":
    SPMVCUDATester().run()