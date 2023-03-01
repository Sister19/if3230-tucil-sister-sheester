# Tugas Kecil - Paralel DFT
Repository untuk kit tugas besar IF3230 Sistem Paralel dan Terdistribusi 2023

### Set Up Environment
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

### How to Run
1. Compile all with this command `make all`
2. Compare serial and parallel with `make run-all file={insert testcase filename}`, e.g. `make run-all file=32`
