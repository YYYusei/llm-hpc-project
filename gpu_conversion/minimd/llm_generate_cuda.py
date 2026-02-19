import sys
sys.path.insert(0, '../../src')
from llm_client import LLMClient

source_code = """
// miniMD Lennard-Jones force calculation
// PAD = 4, MMD_float = double

void compute_fullneigh(int nlocal, int ntypes,
    const double* x, double* f, const int* type,
    const int* neighbors, const int* numneigh, int maxneighs,
    const double* cutforcesq, const double* epsilon, const double* sigma6)
{
    for(int i = 0; i < nlocal; i++) {
        f[i*PAD+0] = f[i*PAD+1] = f[i*PAD+2] = 0.0;
    }
    
    for(int i = 0; i < nlocal; i++) {
        double xtmp = x[i*PAD+0], ytmp = x[i*PAD+1], ztmp = x[i*PAD+2];
        int type_i = type[i];
        double fix = 0, fiy = 0, fiz = 0;
        
        for(int k = 0; k < numneigh[i]; k++) {
            int j = neighbors[i*maxneighs + k];
            double delx = xtmp - x[j*PAD+0];
            double dely = ytmp - x[j*PAD+1];
            double delz = ztmp - x[j*PAD+2];
            double rsq = delx*delx + dely*dely + delz*delz;
            int type_ij = type_i * ntypes + type[j];
            
            if(rsq < cutforcesq[type_ij]) {
                double sr2 = 1.0 / rsq;
                double sr6 = sr2 * sr2 * sr2 * sigma6[type_ij];
                double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon[type_ij];
                fix += delx * force;
                fiy += dely * force;
                fiz += delz * force;
            }
        }
        f[i*PAD+0] = fix;
        f[i*PAD+1] = fiy;
        f[i*PAD+2] = fiz;
    }
}
"""

prompt = f"""Convert the following CPU code to CUDA kernel.

Requirements:
1. Create a __global__ kernel function with one thread per atom
2. Use double precision
3. PAD = 4
4. Output ONLY the CUDA kernel code, no explanations

CPU Code:
```cpp
{source_code}
```

Output the CUDA __global__ kernel:"""

client = LLMClient()
print(f"Using model: {client.model}")
print("Generating CUDA kernel...")

response = client.chat(prompt)

print("\n" + "="*50)
print("LLM Generated CUDA Kernel:")
print("="*50)
print(response.content)

# 保存结果
with open("force_lj_llm_v1.cu", "w") as f:
    f.write(response.content)

print(f"\nSaved to: force_lj_llm_v1.cu")
print(f"Tokens: {response.total_tokens}")