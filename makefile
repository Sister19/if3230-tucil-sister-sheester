OUTPUT_FOLDER = bin

all: serial parallel

mpi:
	mpicc src/open-mpi/mpi.c -o $(OUTPUT_FOLDER)/mpi -lm

runmpi:
	time mpirun -n 4 ./bin/mpi < test_case/$(file).txt > output/mpioutput.txt

serial:
	gcc src/serial/c/serial.c -o $(OUTPUT_FOLDER)/serial -lm

runserial:
	time bin/serial < test_case/$(file).txt > output/serialoutput.txt