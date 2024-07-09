# Navier-Stokes project
This project focuses on simulating incompressible laminar flow around cylinders using the finite element method implemented using deal.II library coded in C++. 
Various preconditioners and MPI for parallelization are employed to improve performance and manage the extensive computational demands. 

## Mesh
The 2D and 3D meshes generated with Gmsh for the simulations are located in the mesh folder. 
For 2D simulations, the file "mesh_2D_coarse" can be used. 
For 3D simulations, a variety of meshes are available, ranging from the coarser "mesh-0.1" to the finer "mesh-0.0125."

## Src
The source code is organized into three main files:

- main.cpp: This file handles the general execution of the program. It serves as the entry point, calling functions, and managing the overall process.
- Navier_Stokes.cpp: This file contains the implementation of all functions needed to assemble and solve the Navier-Stokes equations.
- Navier_Stokes.hpp: This header file contains the declarations of all the classes and function prototypes used in the project. It defines the data structures and interfaces ecessary for the implementation in Navier_Stokes.cpp.

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
A helper function can be summoned by writing
```
./executable-name -h
```
or

```
./executable-name --help
```
