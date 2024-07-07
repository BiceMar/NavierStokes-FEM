# nmpde-project3
Implementation of a finite element solver for the unsteady, incompressible Navier-Stokes equations to simulate the 2D or 3D benchmark problem “flow past a cylinder”
for different values of the Reynolds number Re ≤ 200.

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
