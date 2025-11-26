<div align="center">

# 🔥 HP-JAX  
### **High-Performance JAX Compute Framework for Scientific Computing, Distributed Systems, and TPU/Pi Clusters**

**Matrix Algebra • Auto-Diff • Distributed HPC • Raspberry Pi Cluster • TPU Acceleration**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![JAX](https://img.shields.io/badge/JAX-0.4%2B-green.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)]()
[![Status](https://img.shields.io/badge/Status-Alpha-red.svg)]()

</div>

---

# ✨ Overview

**HP-JAX** is a **high-performance scientific computing framework** built on top of **Google JAX**,  
designed for **matrix computations**, **automatic differentiation**, and **distributed linear algebra**  
on **CPU**, **Raspberry Pi clusters**, and **TPU**.

It aims to bridge the gap between:

- ⚡ *High-performance numerical linear algebra*  
- 🔬 *Scientific computing & optimization*  
- 🧠 *JAX auto-diff (grad/Jacobian/Hessian)*  
- 🔗 *Distributed HPC (MPI)*  
- 🧩 *Low-cost clusters (Raspberry Pi & ARM64)*  
- 🚀 *Future: Cloud TPU acceleration*  

> **HP-JAX = SciPy + JAX + Distributed HPC, but lightweight and fully differentiable.**

GPU support will be added later — current focus is **CPU, ARM, MPI, and TPU**.

---

# 🔥 Features

### 🧮 **Matrix Algebra (CPU / Pi / TPU)**
- `matmul`, `inverse`, `det`, `transpose`
- Eigen decomposition (`eig`)
- SVD (`svd`)
- Cholesky (`cholesky`)
- QR decomposition (coming in v0.2)
- LU decomposition (coming in v0.2)
- Hessenberg / Schur (coming in v0.2)

### 🧠 **Automatic Differentiation**
- `gradient(f)`
- `jacobian(f)`
- `hessian(f)`
- Directional derivatives

Everything is **JIT-accelerated** and fully differentiable via JAX.

### 🧩 **Matrix Partitioning**
- 1-D row split (v0.1)
- 2-D block partition (v0.2)
- Ready for distributed workloads

### 🌐 **Distributed Linear Algebra (MPI)**
> **Optional — only installed if needed.**  
