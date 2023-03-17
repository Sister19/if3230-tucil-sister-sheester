# Tugas Kecil - Paralel DFT
Repository untuk kit tugas besar IF3230 Sistem Paralel dan Terdistribusi 2023

### Set Up Environment (For OpenMPI and OpenMP)
1. Use WSL or Linux environment and open terminal
2. Update the Linux with this command
```
sudo apt-get update
sudo apt-get upgrade
```
3. If you don't have C/C++ compiler, install with this command
```
sudo apt-get install build-essential
```

### Set Up Environment (For Cuda)
1. Use Google Collab/Jupyter Notebook

### How to Run (For OpenMPI and OpenMP)
1. Compile all with this command `make all`
2. Compare serial and parallel with `make run-all file={insert testcase filename}`, e.g. `make run-all file=32`
3. If you want to run only OpenMPI program, run with `make runmpi file={insert testcase filename}`, e.g. `make runmpi file=32`
4. If you want to run only OpenMP program, run with `make runmp file={insert testcase filename}`, e.g. `make runmp file=32`

### How to Run (For OpenMPI and OpenMP)
1. Compile and run inside the Google Collab/Jupyter Notebook
