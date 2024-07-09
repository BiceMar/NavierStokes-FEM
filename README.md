# Navier-Stokes project
This project focuses on simulating incompressible laminar flow around cylinders using the finite element method implemented using deal.II library coded in C++. 
Various preconditioners and MPI for parallelization are utilized to improve performance and manage the extensive computational demands. 

# Compiling

To build the executable, make sure you have loaded the needed modules with
```
module load gcc-glibc dealii
```
Then run the following commands:
```
mkdir build
cd build
cmake ..
make
```
The executable will be created into build, and can be executed through ./executable-name.

By executing only ./executable-main, the program will be executed with default paramenters;
to change the parameters of the simulation run:
```
./executable-name --flag <value>
```
a helper function can be summoned by writing
```
./executable-name -h
```
or

```
./executable-name --help
```
