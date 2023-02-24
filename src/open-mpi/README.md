Cara Kerja Parelesisasi Program OPEN-MPI
- Pertama, process 0 dari program akan membaca input matriks dan membagikan semua data matriks tersebut ke process-process lain.
- Kedua, Akan dilakukan looping dft seperti serial, process 1 akan mengolah fungsi dft pertama, process 2 akan mengolah fungsi dft kedua, process 3 akan mengolah fungsi dft ketiga, lalu process 1 akan melanjutkan mengolah fungsi dft ke 4, dan seterusnya.
- Ketiga, setiap kali selesai mengolah dft, process tersebut akan mengirimnya ke process 0 untuk disimpan.
- Keempat, seteleha process 1-3 selesai, maka paralelisasi selesai dan process 0 melanjutkan untuk melakukan output file

Cara Pembagian Data
- Untuk data matriks, setiap process memerlukan data tersebut sehingga semua process harus memiliki kopi data matriks tersebut.
- Untuk data freq_matriks (hasil dft), hanya process 0 saja yang memerlukan sehingga hanya dibuat dan disimpan di process 0