## Cara Kerja Paralelisasi OpenMP
- Program akan berjalan dengan multi-thread dan mengerjakan setiap thread secara paralel
- OpenMP dengan command `#pragma omp parallel for` akan membuka proses multi-thread dan mengerjakan program secara multi-thread

## Cara Pembagian Data
- Terdapat 2 jenis data yang kami gunakan pada thread, private dan reduction variable
- Private variable akan membuat copy dari variable ke salah satu thread dan akan dikerjakan secara paralel, seperti variable `el` untuk menyimpan elemen matriks
- Reduction variable akan membuat variable yang dapat digunakan secara paralel pada semua thread, seperti variable `sum` untuk menyimpan hasil perhitungan total matriks
