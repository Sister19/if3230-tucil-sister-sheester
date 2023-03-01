OUTPUT_FOLDER = bin
hostfile = util/k01-06

all: serial parallel

run-all: runserial runmpi runmp

parallel: mpi mp

serial:
	gcc src/serial/c/serial.c -o $(OUTPUT_FOLDER)/serial -lm

runserial:
	time bin/serial < test_case/$(file).txt > output/serialoutput$(file).txt

mpi:
	mpicc src/open-mpi/mpi.c -o $(OUTPUT_FOLDER)/mpi -lm

runmpi:
	time mpirun -n 4 bin/mpi < test_case/$(file).txt > output/mpioutput$(file).txt

runmpihost:
	time mpirun --hostfile $(hostfile) bin/mpi < test_case/$(file).txt > output/mpioutputhost$(file).txt

mp:
	gcc src/open-mp/mp.c --openmp -o $(OUTPUT_FOLDER)/mp -lm

runmp:
	time bin/mp < test_case/$(file).txt > output/mpoutput$(file).txt