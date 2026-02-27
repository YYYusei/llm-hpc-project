"""
SYMGS Full Pipeline Test:
1. Cascaded Analysis (GPT-4o → GPT-5.2) to get optimization suggestions
2. Use suggestions to generate CUDA code
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

SYMGS_CODE = '''
int ComputeSYMGS_ref(const SparseMatrix & A, const Vector & r, Vector & x) {
  const local_int_t nrow = A.localNumberOfRows;
  double ** matrixDiagonal = A.matrixDiagonal;
  const double * const rv = r.values;
  double * const xv = x.values;

  // Forward sweep
  for (local_int_t i=0; i < nrow; i++) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0];
    double sum = rv[i];
    for (int j=0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i] * currentDiagonal;
    xv[i] = sum / currentDiagonal;
  }

  // Backward sweep
  for (local_int_t i=nrow-1; i >= 0; i--) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double currentDiagonal = matrixDiagonal[i][0];
    double sum = rv[i];
    for (int j=0; j < currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i] * currentDiagonal;
    xv[i] = sum / currentDiagonal;
  }
  return 0;
}
'''

BENCHMARK_TEMPLATE = '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) {{ cudaError_t e=call; if(e!=cudaSuccess){{printf("CUDA_ERROR: %s\\n",cudaGetErrorString(e));exit(1);}} }}

// ============ LLM Generated SYMGS ============
{optimized_kernel}

// ============ CPU Reference ============
void symgs_cpu(int nrow, int max_nnz, const int* nnz_per_row,
               const int* col_ind, const double* values, const double* diag,
               const double* r, double* x) {{
    for (int i = 0; i < nrow; i++) {{
        double sum = r[i];
        for (int j = 0; j < nnz_per_row[i]; j++) {{
            int idx = i * max_nnz + j;
            sum -= values[idx] * x[col_ind[idx]];
        }}
        sum += x[i] * diag[i];
        x[i] = sum / diag[i];
    }}
    for (int i = nrow - 1; i >= 0; i--) {{
        double sum = r[i];
        for (int j = 0; j < nnz_per_row[i]; j++) {{
            int idx = i * max_nnz + j;
            sum -= values[idx] * x[col_ind[idx]];
        }}
        sum += x[i] * diag[i];
        x[i] = sum / diag[i];
    }}
}}

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
    int num_colors = 8;
    
    int* h_nnz_per_row = (int*)malloc(nrow * sizeof(int));
    int* h_col_ind = (int*)malloc(nrow * max_nnz * sizeof(int));
    double* h_values = (double*)malloc(nrow * max_nnz * sizeof(double));
    double* h_diag = (double*)malloc(nrow * sizeof(double));
    double* h_r = (double*)malloc(nrow * sizeof(double));
    double* h_x_cpu = (double*)malloc(nrow * sizeof(double));
    double* h_x_gpu = (double*)malloc(nrow * sizeof(double));
    int* h_row_colors = (int*)malloc(nrow * sizeof(int));
    
    compute_colors(nrow, h_row_colors, num_colors);
    
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
    
    clock_t cpu_start = clock();
    for (int iter = 0; iter < 3; iter++) {{
        for (int i = 0; i < nrow; i++) h_x_cpu[i] = 0.0;
        symgs_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_diag, h_r, h_x_cpu);
    }}
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0 / 3.0;
    
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
    
    for (int i = 0; i < nrow; i++) h_x_cpu[i] = 0.0;
    symgs_cpu(nrow, max_nnz, h_nnz_per_row, h_col_ind, h_values, h_diag, h_r, h_x_cpu);
    double err = check_correctness(h_x_cpu, h_x_gpu, nrow);
    
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\\n",
           cpu_ms, gpu_ms, cpu_ms/gpu_ms, err);
    
    return 0;
}}
'''


class FullPipelineTester:
    def __init__(self, output_dir: str = "results/cuda_symgs_pipeline"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.clients = {
            "gpt-4o": LLMClient(model="gpt-4o"),
            "gpt-5.2": LLMClient(model="gpt-5.2")
        }
        self.total_cost = 0
    
    def stage1_analysis(self) -> Dict[str, Any]:
        """Stage 1: GPT-4o analyzes the code"""
        logger.info("[Stage 1] GPT-4o analyzing SYMGS code...")
        
        prompt = f'''
Analyze this HPC code for GPU parallelization:
```cpp
{SYMGS_CODE}
```

Identify:
1. Performance bottlenecks
2. Data dependencies that prevent parallelization
3. Recommended GPU parallelization strategy

Output JSON:
```json
{{
    "bottleneck_type": "...",
    "data_dependencies": ["..."],
    "parallelization_challenges": ["..."],
    "recommended_strategy": "...",
    "gpu_suitable": true/false
}}
```
'''
        response = self.clients["gpt-4o"].chat(prompt, system_prompt="You are an HPC expert.")
        self.total_cost += response.cost
        
        return {
            "model": "gpt-4o",
            "response": response.content,
            "cost": response.cost,
            "time": response.elapsed_time
        }
    
    def stage2_deep_analysis(self, stage1_result: str) -> Dict[str, Any]:
        """Stage 2: GPT-5.2 validates and provides detailed strategy"""
        logger.info("[Stage 2] GPT-5.2 providing detailed parallelization strategy...")
        
        prompt = f'''
Review this analysis of SYMGS (Symmetric Gauss-Seidel) and provide a DETAILED GPU implementation strategy.

## Previous Analysis:
{stage1_result}

## Original Code:
```cpp
{SYMGS_CODE}
```

Provide a CONCRETE parallelization strategy for GPU. Consider:
1. How to handle the row-to-row data dependencies
2. Multi-coloring approach (if applicable)
3. Specific CUDA kernel design

Output JSON:
```json
{{
    "validation": "correct/incorrect/partial",
    "corrected_analysis": "...",
    "gpu_strategy": {{
        "approach": "multi-coloring / level-scheduling / block-jacobi / other",
        "description": "detailed description",
        "num_colors_or_levels": 8,
        "kernel_design": "one kernel per color, each thread handles one row",
        "data_structures_needed": ["row_colors array", "..."]
    }},
    "expected_speedup": "2-5x",
    "implementation_notes": ["note1", "note2"]
}}
```
'''
        response = self.clients["gpt-5.2"].chat(prompt, system_prompt="You are a GPU parallelization expert.")
        self.total_cost += response.cost
        
        return {
            "model": "gpt-5.2",
            "response": response.content,
            "cost": response.cost,
            "time": response.elapsed_time
        }
    
    def stage3_generate_cuda(self, stage2_result: str, func_name: str) -> Dict[str, Any]:
        """Stage 3: Generate CUDA code based on the strategy"""
        logger.info("[Stage 3] Generating CUDA code based on analysis...")
        
        prompt = f'''
Based on this parallelization strategy, generate CUDA code for SYMGS:

## Strategy:
{stage2_result}

## Requirements:
1. Use multi-coloring approach
2. Host function signature MUST be:
```cuda
   void {func_name}(int nrow, int max_nnz, int num_colors,
       const int* row_colors, const int* nnz_per_row, const int* col_ind,
       const double* values, const double* diag, const double* r, double* x)
```
3. Process forward sweep (colors 0 to num_colors-1), then backward sweep (reverse order)
4. Each kernel processes rows of ONE color in parallel

## Data Format:
- row_colors[i]: color of row i (0 to num_colors-1)
- nnz_per_row[i]: number of non-zeros in row i
- col_ind[i * max_nnz + j]: column index
- values[i * max_nnz + j]: matrix value  
- diag[i]: diagonal element
- r[]: RHS, x[]: solution (in/out)

Output ONLY the CUDA code:
```cuda
// kernel + host function
```
'''
        response = self.clients["gpt-5.2"].chat(prompt, system_prompt="Generate clean, working CUDA code.")
        self.total_cost += response.cost
        
        # Extract code
        code = response.content
        for pattern in [r'```cuda\n(.*?)```', r'```cpp\n(.*?)```', r'```\n(.*?)```']:
            match = re.search(pattern, code, re.DOTALL)
            if match:
                code = match.group(1).strip()
                break
        
        return {
            "model": "gpt-5.2",
            "code": code,
            "cost": response.cost,
            "time": response.elapsed_time
        }
    
    def compile_and_test(self, code: str, func_name: str, test_name: str) -> Tuple[bool, str, Optional[Dict]]:
        """Compile and benchmark"""
        logger.info(f"[{test_name}] Compiling and testing...")
        
        benchmark_code = BENCHMARK_TEMPLATE.format(optimized_kernel=code, func_name=func_name)
        
        cu_file = os.path.join(self.output_dir, f"{test_name}.cu")
        with open(cu_file, 'w') as f:
            f.write(benchmark_code)
        
        win_path = os.path.abspath(self.output_dir)
        wsl_path = "/mnt/" + win_path[0].lower() + win_path[2:].replace("\\", "/")
        
        compile_cmd = f'wsl -d Ubuntu-24.04 bash -c "cd {wsl_path} && /usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_86 {test_name}.cu -o {test_name} 2>&1"'
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            error = result.stdout + result.stderr
            logger.warning(f"Compilation failed: {error[:200]}")
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
    
    def fix_code(self, code: str, error: str, func_name: str) -> str:
        """Fix compilation errors"""
        logger.info("[Fix] GPT-5.2 fixing code...")
        
        prompt = f'''
Fix this CUDA code:
```cuda
{code}
```

Error: {error[:500]}

Host function MUST be:
void {func_name}(int nrow, int max_nnz, int num_colors,
    const int* row_colors, const int* nnz_per_row, const int* col_ind,
    const double* values, const double* diag, const double* r, double* x)

Output ONLY fixed code:
```cuda
```
'''
        response = self.clients["gpt-5.2"].chat(prompt)
        self.total_cost += response.cost
        
        code = response.content
        for pattern in [r'```cuda\n(.*?)```', r'```cpp\n(.*?)```', r'```\n(.*?)```']:
            match = re.search(pattern, code, re.DOTALL)
            if match:
                return match.group(1).strip()
        return code
    
    def run(self):
        """Run full pipeline"""
        logger.info("=" * 60)
        logger.info("SYMGS Full Pipeline: Cascaded Analysis → CUDA Generation")
        logger.info("=" * 60)
        
        results = {"timestamp": datetime.now().isoformat(), "stages": {}}
        func_name = "symgs_gpu_pipeline"
        
        # Stage 1: GPT-4o Analysis
        stage1 = self.stage1_analysis()
        results["stages"]["stage1_analysis"] = stage1
        logger.info(f"[Stage 1] Complete. Cost: ${stage1['cost']:.4f}")
        
        # Stage 2: GPT-5.2 Deep Analysis
        stage2 = self.stage2_deep_analysis(stage1["response"])
        results["stages"]["stage2_deep_analysis"] = stage2
        logger.info(f"[Stage 2] Complete. Cost: ${stage2['cost']:.4f}")
        
        # Stage 3: Generate CUDA
        stage3 = self.stage3_generate_cuda(stage2["response"], func_name)
        results["stages"]["stage3_cuda_generation"] = stage3
        logger.info(f"[Stage 3] Complete. Cost: ${stage3['cost']:.4f}")
        
        with open(os.path.join(self.output_dir, "generated_code.cu"), 'w') as f:
            f.write(stage3["code"])
        
        # Test
        success, error, benchmark = self.compile_and_test(stage3["code"], func_name, "pipeline_v1")
        
        if not success:
            logger.info("[Fix] Attempting to fix...")
            fixed_code = self.fix_code(stage3["code"], error, func_name)
            with open(os.path.join(self.output_dir, "fixed_code.cu"), 'w') as f:
                f.write(fixed_code)
            success, error, benchmark = self.compile_and_test(fixed_code, func_name, "pipeline_v2")
        
        results["final_test"] = {
            "success": success,
            "benchmark": benchmark,
            "error": error if not success else None
        }
        results["total_cost"] = self.total_cost
        
        with open(os.path.join(self.output_dir, "pipeline_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary
        print("\n" + "=" * 70)
        print("SYMGS FULL PIPELINE RESULTS")
        print("=" * 70)
        print(f"Stage 1 (GPT-4o Analysis): ${stage1['cost']:.4f}")
        print(f"Stage 2 (GPT-5.2 Strategy): ${stage2['cost']:.4f}")
        print(f"Stage 3 (CUDA Generation): ${stage3['cost']:.4f}")
        print(f"Total Cost: ${self.total_cost:.4f}")
        print()
        if success and benchmark:
            print(f"Result: SUCCESS")
            print(f"  CPU: {benchmark['cpu_ms']:.2f}ms")
            print(f"  GPU: {benchmark['gpu_ms']:.2f}ms")
            print(f"  Speedup: {benchmark['speedup']:.2f}x")
            print(f"  Error: {benchmark['error']:.2e}")
        else:
            print(f"Result: FAILED")
            print(f"  Error: {error[:200]}")
        print("=" * 70)
        
        return results


if __name__ == "__main__":
    FullPipelineTester().run()