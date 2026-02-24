import sys
import json
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, 'src')

from llm_client import LLMClient

# Existing baseline kernel
BASELINE_KERNEL = '''
__global__ void compute_fullneigh_kernel(int nlocal, int ntypes,
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
Please optimize the following CUDA kernel for Lennard-Jones force calculation in molecular dynamics simulation.

## Current Baseline Kernel:
```cuda
{baseline}
```

## Performance Data:
- Current speedup: 10-13x (vs CPU)
- GPU: RTX 3060 (SM 8.6, 12GB)
- Typical problem size: 50,000 - 200,000 atoms
- Each atom has ~30-50 neighbors

## Please apply the following optimization techniques (where applicable):
1. **Shared Memory**: Cache frequently accessed data
2. **Memory Coalescing**: Optimize global memory access patterns
3. **Loop Unrolling**: Reduce loop overhead
4. **Register Optimization**: Reduce register pressure
5. **Warp-level Optimization**: Utilize warp-level cooperation

## Requirements:
1. Maintain numerical correctness
2. Output complete compilable optimized kernel
3. Add comments to mark each optimization point

## Output format:
```cuda
// Optimized kernel code here
```
'''


def test_optimization(model: str) -> dict:
    """Test optimization capability of a single model"""
    print(f"\n{'='*50}")
    print(f"Testing model: {model}")
    print('='*50)
    
    client = LLMClient(model=model)
    
    prompt = OPTIMIZATION_PROMPT.format(baseline=BASELINE_KERNEL)
    
    response = client.chat(
        prompt=prompt,
        system_prompt="You are a CUDA performance optimization expert. Generate high-quality, compilable optimized code."
    )
    
    print(f"Elapsed time: {response.elapsed_time:.2f}s")
    print(f"Cost: ${response.cost:.4f}")
    print(f"Tokens: {response.total_tokens}")
    
    # Extract code
    code_match = re.search(r'```cuda\n(.*?)```', response.content, re.DOTALL)
    if not code_match:
        code_match = re.search(r'```cpp\n(.*?)```', response.content, re.DOTALL)
    if not code_match:
        code_match = re.search(r'```\n(.*?)```', response.content, re.DOTALL)
    
    optimized_code = code_match.group(1) if code_match else response.content
    
    return {
        "model": model,
        "optimized_code": optimized_code,
        "cost": response.cost,
        "elapsed_time": response.elapsed_time,
        "total_tokens": response.total_tokens,
        "raw_response": response.content
    }


def main():
    results = {}
    
    # Test both models
    for model in ["gpt-4o", "gpt-5.2"]:
        results[model] = test_optimization(model)
    
    # Save results
    os.makedirs("results/cuda_optimization", exist_ok=True)
    
    for model, result in results.items():
        # Save code
        filename = f"results/cuda_optimization/{model.replace('.', '_')}_optimized.cu"
        with open(filename, 'w') as f:
            f.write(result['optimized_code'])
        print(f"\n{model} optimized code saved to {filename}")
    
    # Save full results
    with open("results/cuda_optimization/optimization_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Print summary
    print("\n" + "="*50)
    print("Optimization Results Summary")
    print("="*50)
    
    for model, result in results.items():
        print(f"\n{model}:")
        print(f"  Code length: {len(result['optimized_code'])} chars")
        print(f"  Cost: ${result['cost']:.4f}")
        print(f"  Time: {result['elapsed_time']:.2f}s")
    
    total_cost = sum(r['cost'] for r in results.values())
    print(f"\nTotal cost: ${total_cost:.4f}")
    print("\nNext step: Compile and benchmark the optimized kernels")


if __name__ == "__main__":
    main()