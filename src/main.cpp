#include "NavierStokes.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const std::string  mesh_file_name  = "../mesh/mesh-0.1.msh";
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;

  const double T      = 0.5;
  const double deltat = 0.05;
  const double theta  = 1.0;

  NavierStokes problem(mesh_file_name, degree_velocity, degree_pressure, T, deltat, theta);

  problem.setup();
  problem.solve();
  
  //.output();

  return 0;
}
