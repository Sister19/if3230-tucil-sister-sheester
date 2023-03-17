## Cara Kerja Paralelisasi CUDA
- Program berjalan dengan menggunakan multi-core GPU dan setiap thread di dalam kernel menjalankan operasi yang sama pada setiap blok matriks yang telah didefinisikan.
- Awalnya, matriks yang akan diproses disalin dari host ke device menggunakan cudaMemcpy().
- Kemudian, dft_kernel() dijalankan sebanyak source.size blok dengan setiap blok memiliki source.size thread.
- Setiap thread akan memuat blok matriks yang dihasilkan dari pembagian ukuran matriks dengan BLOCK_SIZE ke dalam shared memory dan melakukan operasi DFT pada blok tersebut.
- Setelah selesai melakukan operasi pada blok, hasil yang dihasilkan dalam bentuk matriks frekuensi disalin kembali dari device ke host menggunakan cudaMemcpy().
- Kemudian, hasil diproses pada host untuk menampilkan matriks frekuensi dan menghitung jumlah keseluruhan elemen dari matriks frekuensi.

## Cara Pembagian Data
- Pembagian data antar thread dilakukan dalam satu blok pada setiap iterasi kernel.
- Pada perhitungan dft, terdapat shared memory yang digunakan untuk menyimpan data input yang dibutuhkan oleh thread pada setiap blok.
- Setiap thread pada blok akan mengakses data pada shared memory tersebut dan melakukan komputasi dengan menggunakan data tersebut.
- Pembagian data menjadi blok-blok dapat mempercepat proses komputasi.
- Penggunaan shared memory dapat mempercepat proses komputasi juga karena mengurangi akses ke memori global.