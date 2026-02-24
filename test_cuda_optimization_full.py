"""
Full CUDA Optimization Test Pipeline
1. Generate optimized code from both models
2. Compile and test
3. If error, let GPT-5.2 fix it (newest model)
4. Benchmark and compare results (CPU vs GPU)
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

BASELINE_KERNEL = '''
__global__ void baseline_kernel(int nlocal, int ntypes,
    const double* __restrict__ x, double* __restrict__ f, const int* __restrict__ type,
    const int* __restrict__ neighbors, const int* __restrict__ numneigh, int maxneighs,
    const double* __restrict__ cutforcesq, const double* __restrict__ epsilon,
    const double* __restrict__ sigma6) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlocal) return;
    double xtmp=x[i*4+0], ytmp=x[i*4+1], ztmp=x[i*4+2];
    int type_i=type[i]; 
    double fix=0, fiy=0, fiz=0;
    int nneigh=numneigh[i], base=i*maxneighs;
    for(int k=0; k<nneigh; k++) {
        int j=neighbors[base+k];
        double delx=xtmp-x[j*4+0], dely=ytmp-x[j*4+1], delz=ztmp-x[j*4+2];
        double rsq=delx*delx+dely*dely+delz*delz;
        int type_ij=type_i*ntypes+type[j];
        if(rsq<cutforcesq[type_ij]) {
            double sr2=1.0/rsq, sr6=sr2*sr2*sr2*sigma6[type_ij];
            double force=48.0*sr6*(sr6-0.5)*sr2*epsilon[type_ij];
            fix+=delx*force; fiy+=dely*force; fiz+=delz*force;
        }
    }
    f[i*4+0]=fix; f[i*4+1]=fiy; f[i*4+2]=fiz;
}
'''

OPTIMIZATION_PROMPT = '''
Please optimize the following CUDA kernel for Lennard-Jones force calculation.

IMPORTANT: Do NOT define ldg_int() or ldg_double() - they are already provided.

## Baseline Kernel:
```cuda
{baseline}
```

## Info:
- GPU: RTX 3060 (SM 8.6)
- Problem size: 50,000 - 200,000 atoms
- Each atom has ~30-50 neighbors

## Apply optimizations:
1. Memory Coalescing (double4 for x/f access)
2. Loop Unrolling
3. FMA instructions (use fma())
4. Read-only cache (__ldg for scalar types)

## Requirements:
1. Maintain numerical correctness
2. Function name must be: {func_name}
3. Do NOT use shared memory for neighbor data
4. Output ONLY the kernel code
```cuda
// Your optimized kernel here
```
'''

FIX_PROMPT = '''
The following CUDA kernel has a bug.

## Buggy Code:
```cuda
{buggy_code}
```

## Error:
{error_msg}

IMPORTANT:
- Do NOT define ldg_int() or ldg_double() - they already exist.
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

#define PAD 4
#define CHECK_CUDA(call) {{ cudaError_t e=call; if(e!=cudaSuccess){{printf("CUDA_ERROR: %s\\n",cudaGetErrorString(e));exit(1);}} }}

// Pre-defined helper functions
__device__ __forceinline__ int ldg_int(const int* p) {{ return __ldg(p); }}
__device__ __forceinline__ double ldg_double(const double* p) {{ return __ldg(p); }}

// ============ Baseline Kernel ============
{baseline_kernel}

// ============ Optimized Kernel ============
{optimized_kernel}

// ============ CPU Reference ============
void cpu_force(int nlocal, int ntypes, const double* x, double* f, const int* type,
    const int* neighbors, const int* numneigh, int maxneighs,
    const double* cutforcesq, const double* epsilon, const double* sigma6) {{
    for(int i = 0; i < nlocal; i++) {{
        double xtmp=x[i*PAD+0],ytmp=x[i*PAD+1],ztmp=x[i*PAD+2];
        int type_i=type[i]; double fix=0,fiy=0,fiz=0;
        for(int k = 0; k < numneigh[i]; k++) {{
            int j=neighbors[i*maxneighs+k];
            double delx=xtmp-x[j*PAD+0],dely=ytmp-x[j*PAD+1],delz=ztmp-x[j*PAD+2];
            double rsq=delx*delx+dely*dely+delz*delz;
            int type_ij=type_i*ntypes+type[j];
            if(rsq<cutforcesq[type_ij]) {{
                double sr2=1.0/rsq,sr6=sr2*sr2*sr2*sigma6[type_ij];
                double force=48.0*sr6*(sr6-0.5)*sr2*epsilon[type_ij];
                fix+=delx*force; fiy+=dely*force; fiz+=delz*force;
            }}
        }}
        f[i*PAD+0]=fix; f[i*PAD+1]=fiy; f[i*PAD+2]=fiz;
    }}
}}

double check_correctness(double* f1, double* f2, int n) {{
    double maxerr = 0.0;
    for(int i = 0; i < n; i++) {{
        for(int c = 0; c < 3; c++) {{
            double err = fabs(f1[i*PAD+c] - f2[i*PAD+c]);
            if(err > maxerr) maxerr = err;
        }}
    }}
    return maxerr;
}}

int main() {{
    int nlocal = 100000;
    int nall = nlocal + nlocal/10;
    int ntypes = 2, maxneighs = 100;
    
    double *h_x = (double*)malloc(nall * PAD * sizeof(double));
    double *h_f_cpu = (double*)malloc(nlocal * PAD * sizeof(double));
    double *h_f_gpu = (double*)malloc(nlocal * PAD * sizeof(double));
    int *h_type = (int*)malloc(nall * sizeof(int));
    int *h_neigh = (int*)malloc(nlocal * maxneighs * sizeof(int));
    int *h_numn = (int*)malloc(nlocal * sizeof(int));
    double *h_cut = (double*)malloc(ntypes * ntypes * sizeof(double));
    double *h_eps = (double*)malloc(ntypes * ntypes * sizeof(double));
    double *h_sig = (double*)malloc(ntypes * ntypes * sizeof(double));
    
    srand(12345);
    for(int i = 0; i < nall; i++) {{
        h_x[i*PAD+0] = (double)rand()/RAND_MAX * 100;
        h_x[i*PAD+1] = (double)rand()/RAND_MAX * 100;
        h_x[i*PAD+2] = (double)rand()/RAND_MAX * 100;
        h_x[i*PAD+3] = 0;
        h_type[i] = rand() % ntypes;
    }}
    for(int i = 0; i < nlocal; i++) {{
        h_numn[i] = 30 + rand() % 20;
        for(int k = 0; k < h_numn[i]; k++)
            h_neigh[i*maxneighs+k] = rand() % nall;
    }}
    for(int i = 0; i < ntypes*ntypes; i++) {{
        h_cut[i] = 6.25; h_eps[i] = 1.0; h_sig[i] = 1.0;
    }}
    
    // CPU benchmark
    clock_t cpu_start = clock();
    for(int r = 0; r < 5; r++)
        cpu_force(nlocal, ntypes, h_x, h_f_cpu, h_type, h_neigh, h_numn, maxneighs, h_cut, h_eps, h_sig);
    clock_t cpu_end = clock();
    double cpu_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0 / 5.0;
    
    // GPU setup
    double *d_x, *d_f, *d_cut, *d_eps, *d_sig;
    int *d_type, *d_neigh, *d_numn;
    CHECK_CUDA(cudaMalloc(&d_x, nall * PAD * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_f, nlocal * PAD * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_type, nall * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_neigh, nlocal * maxneighs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_numn, nlocal * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cut, ntypes * ntypes * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_eps, ntypes * ntypes * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_sig, ntypes * ntypes * sizeof(double)));
    
    CHECK_CUDA(cudaMemcpy(d_x, h_x, nall * PAD * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_type, h_type, nall * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_neigh, h_neigh, nlocal * maxneighs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_numn, h_numn, nlocal * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cut, h_cut, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_eps, h_eps, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sig, h_sig, ntypes * ntypes * sizeof(double), cudaMemcpyHostToDevice));
    
    int bs = 256, nb = (nlocal + bs - 1) / bs;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    // GPU benchmark (LLM generated kernel)
    {func_name}<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(start);
    for(int r = 0; r < 10; r++)
        {func_name}<<<nb, bs>>>(nlocal, ntypes, d_x, d_f, d_type, d_neigh, d_numn, maxneighs, d_cut, d_eps, d_sig);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float gpu_ms; cudaEventElapsedTime(&gpu_ms, start, stop); gpu_ms /= 10;
    CHECK_CUDA(cudaMemcpy(h_f_gpu, d_f, nlocal * PAD * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Check correctness
    double err = check_correctness(h_f_cpu, h_f_gpu, nlocal);
    
    // Output: CPU vs GPU
    printf("BENCHMARK_RESULT:cpu_ms=%.4f,gpu_ms=%.4f,speedup=%.2f,error=%.2e\\n", 
           cpu_ms, gpu_ms, cpu_ms/gpu_ms, err);
    
    free(h_x); free(h_f_cpu); free(h_f_gpu);
    free(h_type); free(h_neigh); free(h_numn); free(h_cut); free(h_eps); free(h_sig);
    cudaFree(d_x); cudaFree(d_f); cudaFree(d_type); cudaFree(d_neigh);
    cudaFree(d_numn); cudaFree(d_cut); cudaFree(d_eps); cudaFree(d_sig);
    
    return 0;
}}
'''


class CUDAOptimizationTester:
    """Full pipeline for testing LLM CUDA optimization capability"""
    
    def __init__(self, output_dir: str = "results/cuda_optimization"):
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
                code = match.group(1).strip()
                # Remove any ldg_int/ldg_double definitions to avoid redefinition
                code = re.sub(r'__device__\s+__forceinline__\s+int\s+ldg_int\s*\([^)]*\)\s*\{[^}]*\}', '', code)
                code = re.sub(r'__device__\s+__forceinline__\s+double\s+ldg_double\s*\([^)]*\)\s*\{[^}]*\}', '', code)
                return code.strip()
        return response.strip()
    
    def generate_optimized_kernel(self, model: str, func_name: str) -> Dict[str, Any]:
        """Step 1: Generate optimized kernel"""
        logger.info(f"[{model}] Generating optimized kernel...")
        
        prompt = OPTIMIZATION_PROMPT.format(
            baseline=BASELINE_KERNEL,
            func_name=func_name
        )
        
        response = self.clients[model].chat(
            prompt=prompt,
            system_prompt="You are a CUDA performance optimization expert. Generate high-quality, compilable optimized code."
        )
        
        code = self.extract_cuda_code(response.content)
        
        return {
            "code": code,
            "cost": response.cost,
            "time": response.elapsed_time,
            "tokens": response.total_tokens
        }
    
    def compile_and_test(self, kernel_code: str, func_name: str, test_name: str) -> Tuple[bool, str, Optional[Dict]]:
        """Step 2: Compile and run benchmark"""
        logger.info(f"[{test_name}] Compiling and testing...")
        
        benchmark_code = BENCHMARK_TEMPLATE.format(
            baseline_kernel=BASELINE_KERNEL,
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
    
    def fix_kernel(self, buggy_code: str, error_msg: str, fixer_model: str, func_name: str) -> Dict[str, Any]:
        """Step 3: Let GPT-5.2 fix the buggy kernel"""
        logger.info(f"[{fixer_model}] Fixing buggy kernel...")
        
        prompt = FIX_PROMPT.format(
            buggy_code=buggy_code,
            error_msg=error_msg[:500],
            func_name=func_name
        )
        
        response = self.clients[fixer_model].chat(
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
    
    def test_model(self, model: str) -> Dict[str, Any]:
        """Full test pipeline for one model"""
        func_name = f"optimized_kernel_{model.replace('.', '_').replace('-', '_')}"
        fixer_model = "gpt-5.2"  # Always use GPT-5.2 as fixer (newest model)
        
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
        
        # Step 1: Generate
        gen_result = self.generate_optimized_kernel(model, func_name)
        result["generation"] = gen_result
        result["total_cost"] += gen_result["cost"]
        
        code_file = os.path.join(self.output_dir, f"{model.replace('.', '_')}_generated.cu")
        with open(code_file, 'w') as f:
            f.write(gen_result["code"])
        
        # Step 2: First test
        success, error_msg, benchmark = self.compile_and_test(
            gen_result["code"], func_name, f"{model.replace('.', '_')}_v1"
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
        
        # Step 3: Fix with GPT-5.2 (always use newest model)
        logger.info(f"[{model}] First attempt failed, asking {fixer_model} to fix...")
        fix_result = self.fix_kernel(gen_result["code"], error_msg, fixer_model, func_name)
        result["fix_attempt"] = {
            "fixer_model": fixer_model,
            **fix_result
        }
        result["total_cost"] += fix_result["cost"]
        
        fixed_file = os.path.join(self.output_dir, f"{model.replace('.', '_')}_fixed_by_{fixer_model.replace('.', '_')}.cu")
        with open(fixed_file, 'w') as f:
            f.write(fix_result["code"])
        
        # Step 4: Test fixed version
        success, error_msg, benchmark = self.compile_and_test(
            fix_result["code"], func_name, f"{model.replace('.', '_')}_v2_fixed"
        )
        result["final_test"] = {
            "success": success,
            "error": error_msg if not success else None,
            "benchmark": benchmark
        }
        
        if success:
            result["final_status"] = f"SUCCESS_AFTER_FIX_BY_{fixer_model}"
        else:
            result["final_status"] = "FAILED"
        
        return result
    
    def run_full_test(self) -> Dict[str, Any]:
        """Run full test for all models"""
        logger.info("="*60)
        logger.info("Starting Full CUDA Optimization Test Pipeline")
        logger.info("="*60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        for model in ["gpt-4o", "gpt-5.2"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {model}")
            logger.info("="*60)
            
            try:
                model_result = self.test_model(model)
                results["models"][model] = model_result
            except Exception as e:
                logger.error(f"[{model}] Test failed with exception: {e}")
                results["models"][model] = {
                    "model": model,
                    "final_status": "EXCEPTION",
                    "error": str(e)
                }
        
        results_file = os.path.join(self.output_dir, "full_test_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """Print test summary"""
        print("\n" + "="*70)
        print("CUDA OPTIMIZATION TEST SUMMARY (CPU vs LLM-Generated GPU)")
        print("="*70)
        
        for model, data in results["models"].items():
            print(f"\n### {model} ###")
            print(f"  Status: {data.get('final_status', 'UNKNOWN')}")
            
            if data.get("generation"):
                print(f"  Generation: ${data['generation']['cost']:.4f}, {data['generation']['time']:.1f}s")
            
            if data.get("first_test", {}).get("success"):
                bench = data["first_test"]["benchmark"]
                print(f"  First Test: SUCCESS")
                print(f"    CPU: {bench['cpu_ms']:.2f}ms, GPU: {bench['gpu_ms']:.2f}ms")
                print(f"    Speedup: {bench['speedup']:.2f}x")
                print(f"    Error: {bench['error']:.2e}")
            elif data.get("first_test"):
                print(f"  First Test: FAILED")
                if data.get("fix_attempt"):
                    print(f"  Fix by: {data['fix_attempt']['fixer_model']}")
                    print(f"  Fix Cost: ${data['fix_attempt']['cost']:.4f}")
            
            if data.get("final_test", {}).get("benchmark"):
                bench = data["final_test"]["benchmark"]
                print(f"  Final Result:")
                print(f"    CPU: {bench['cpu_ms']:.2f}ms, GPU: {bench['gpu_ms']:.2f}ms")
                print(f"    Speedup: {bench['speedup']:.2f}x")
                print(f"    Error: {bench['error']:.2e}")
            
            print(f"  Total Cost: ${data.get('total_cost', 0):.4f}")
        
        print("\n" + "="*70)
        total_cost = sum(d.get("total_cost", 0) for d in results["models"].values())
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Results saved to: {self.output_dir}/full_test_results.json")
        print("="*70)


def main():
    tester = CUDAOptimizationTester()
    tester.run_full_test()


if __name__ == "__main__":
    main()