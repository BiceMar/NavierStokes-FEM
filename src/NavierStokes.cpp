#include "NavierStokes.hpp"

void
NavierStokes::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
    const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
    fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity,
                                         dim,
                                         fe_scalar_pressure,
                                         1);

    pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree
          << std::endl;
    pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree
          << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

    pcout << "  Quadrature points per face = " << quadrature_face->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We want to reorder DoFs so that all velocity DoFs come first, and then
    // all pressure DoFs.
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Besides the locally owned and locally relevant indices for the whole
    // system (velocity and pressure), we will also need those for the
    // individual velocity and pressure blocks.
    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    block_owned_dofs.resize(2);
    block_relevant_dofs.resize(2);
    block_owned_dofs[0]    = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1]    = locally_owned_dofs.get_view(n_u, n_u + n_p);
    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    pcout << "  Number of DoFs: " << std::endl;
    pcout << "    velocity = " << n_u << std::endl;
    pcout << "    pressure = " << n_p << std::endl;
    pcout << "    total    = " << n_u + n_p << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // Velocity DoFs interact with other velocity DoFs (the weak formulation has
    // terms involving u times v), and pressure DoFs interact with velocity DoFs
    // (there are terms involving p times v or u times q). However, pressure
    // DoFs do not interact with other pressure DoFs (there are no terms
    // involving p times q). We build a table to store this information, so that
    // the sparsity pattern can be built accordingly.
    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      {
        for (unsigned int d = 0; d < dim + 1; ++d)
          {
            if (c == dim && d == dim) // pressure-pressure term
              coupling[c][d] = DoFTools::none;
            else // other combinations
              coupling[c][d] = DoFTools::always;
          }
      }

    TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                    MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
    sparsity.compress();

    // We also build a sparsity pattern for the pressure mass matrix.
    for (unsigned int c = 0; c < dim + 1; ++c)
      {
        for (unsigned int d = 0; d < dim + 1; ++d)
          {
            if (c == dim && d == dim) // pressure-pressure term
              coupling[c][d] = DoFTools::always;
            else // other combinations
              coupling[c][d] = DoFTools::none;
          }
      }
    TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
      block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    coupling,
                                    sparsity_pressure_mass);
    sparsity_pressure_mass.compress();

    pcout << "  Initializing the matrices" << std::endl;

    pressure_mass.reinit(sparsity_pressure_mass);
    system_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  }
}

void
NavierStokes::assemble_time_independent()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling time independent component" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim>     fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_lhs_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_rhs_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  lhs_matrix = 0.0;
  rhs_matrix = 0.0;
  pressure_mass = 0.0;

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_lhs_matrix = 0.0;
      cell_rhs_matrix = 0.0;
      cell_pressure_mass_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  
                  // Viscosity term (A * Theta)
                  cell_lhs_matrix(i, j) +=
                    nu * theta *
                    scalar_product(fe_values[velocity].gradient(i, q),
                                   fe_values[velocity].gradient(j, q)) *
                    fe_values.JxW(q);

                  // Mass Matrix M/deltat
                  cell_lhs_matrix(i, j) += 
                                            scalar_product(fe_values[velocity].value(i, q),
                                                          fe_values[velocity].value(j, q)) 
                                                          / deltat * fe_values.JxW(q);
                    
                  // Pressure term in the momentum equation.
                  cell_lhs_matrix(i, j) -= fe_values[velocity].divergence(i, q) *
                                       fe_values[pressure].value(j, q) *
                                       fe_values.JxW(q);

                  // Pressure term in the continuity equation.
                  cell_lhs_matrix(i, j) -= fe_values[velocity].divergence(j, q) *
                                       fe_values[pressure].value(i, q) *
                                       fe_values.JxW(q);

                  // M/deltaT
                  cell_rhs_matrix(i, j) += 
                    scalar_product(fe_values[velocity].value(i, q),
                                   fe_values[velocity].value(j, q)) /
                    deltat * fe_values.JxW(q);

                  // A*(1-theta)               
                  cell_rhs_matrix(i, j) +=
                    nu * (1.0 - theta) * 
                    scalar_product(fe_values[velocity].gradient(i, q),
                                   fe_values[velocity].gradient(j, q)) *
                    fe_values.JxW(q);

                  // Pressure mass matrix.
                  cell_pressure_mass_matrix(i, j) +=
                    fe_values[pressure].value(i, q) *
                    fe_values[pressure].value(j, q) / nu * fe_values.JxW(q);
                }
        }
      }

      cell->get_dof_indices(dof_indices);

      lhs_matrix.add(dof_indices, cell_lhs_matrix);
      rhs_matrix.add(dof_indices, cell_rhs_matrix);
      pressure_mass.add(dof_indices, cell_pressure_mass_matrix);

    }

  lhs_matrix.compress(VectorOperation::add);
  rhs_matrix.compress(VectorOperation::add);
  pressure_mass.compress(VectorOperation::add);
  
}


void 
NavierStokes::assemble_system()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system time step" << std::endl;
  
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();

  FEValues<dim>     fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  //FullMatrix<double> cell_Fp_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  std::vector<Tensor<1, dim>> velocity_loc(n_q);
  std::vector<Tensor<2, dim>> velocity_gradient_loc(n_q);
  // Declare a vector which will contain the values of the old solution at
  // quadrature points (for the skew-symmetric form of the nonlinear term).
  std::vector<Tensor<1, dim>> old_solution_values(n_q);

  system_matrix = 0.0;
  system_rhs    = 0.0;

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  // Declare tensors to store the old solution value at a set quadrature
  // point and part of the nonlinear term (for the skew-symmetric form of the nonlinear term).
  Tensor<1, dim> local_old_solution_value;
  Tensor<1, dim> nonlinear_term;
  Tensor<1, dim> transpose_term;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_matrix = 0.0;
    cell_rhs    = 0.0;

    fe_values[velocity].get_function_values(solution, velocity_loc);
    fe_values[velocity].get_function_gradients(solution, velocity_gradient_loc);
    // Calculate the value of the previous solution at quadrature points.
    fe_values[velocity].get_function_values(solution, old_solution_values);

    for (unsigned int q = 0; q < n_q; ++q)
    {
      
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          { 
            // Reset the nonlinear and transpose terms.
            nonlinear_term = 0.0;
            transpose_term = 0.0;

            for (unsigned int k = 0; k < dim; k++) {
              for (unsigned int l = 0; l < dim; l++) {
                // Calculate (u . nabla) u.
                nonlinear_term[k] += old_solution_values[q][l] *
                                      fe_values[velocity].gradient(j, q)[k][l];
                // Calculate (nabla u)^T u.
                transpose_term[k] += old_solution_values[q][k] *
                                      fe_values[velocity].gradient(j, q)[l][k];
              }
            }

            // Add the skew-symmetric term (u . nabla) u + (nabla u)^T u to the matrix.
            cell_matrix(i, j) +=
                0.5 * (scalar_product(nonlinear_term, fe_values[velocity].value(i, q)) +
                        scalar_product(transpose_term, fe_values[velocity].value(i, q))) *
                fe_values.JxW(q);             
          }

          Vector<double> forcing_term_new_loc(dim);
          Vector<double> forcing_term_old_loc(dim);

          forcing_term.set_time(time);
          forcing_term.vector_value(fe_values.quadrature_point(q),
                                    forcing_term_new_loc);

          forcing_term.set_time(time - deltat);
          forcing_term.vector_value(fe_values.quadrature_point(q),
                                    forcing_term_old_loc);

          Tensor<1, dim> forcing_term_tensor_new;
          Tensor<1, dim> forcing_term_tensor_old;

          for (unsigned int d = 0; d < dim; ++d)
          {
            forcing_term_tensor_new[d] = forcing_term_new_loc[d];
            forcing_term_tensor_old[d] = forcing_term_old_loc[d];
          }   
          
          // Forcing term.
          cell_rhs(i) += scalar_product(theta  * forcing_term_tensor_new +
                                  (1.0 - theta) * forcing_term_tensor_old,
                                        fe_values[velocity].value(i, q)) *
                        fe_values.JxW(q);
        }
    }

    if (cell->at_boundary())
      {
        for (unsigned int f = 0; f < cell->n_faces(); ++f)
          {
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == 1)
              {
                fe_face_values.reinit(cell, f);

                for (unsigned int q = 0; q < n_q_face; ++q)
                  {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        cell_rhs(i) +=
                          - p_out *
                          scalar_product(fe_face_values.normal_vector(q),
                                        fe_face_values[velocity].value(i,q)) *
                          fe_face_values.JxW(q);
                      }
                  }
              }
          }
      }
      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);     
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  
  // (M/deltaT + A*(1-theta))*u_n + (1-theta)*F(t_n) + theta*F(t_n+1) 
  rhs_matrix.vmult_add(system_rhs, solution_owned);
  system_matrix.add(1.0, lhs_matrix);

  // Dirichlet boundary conditions.
  {
    std::map<types::global_dof_index, double>           boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    Functions::ZeroFunction<dim> zero_function(dim + 1);

    //Inflow boundary condition
    inlet_velocity.set_time(time);
    boundary_functions[0] = &inlet_velocity;
    VectorTools::interpolate_boundary_values(dof_handler,
                                               boundary_functions,
                                               boundary_values,
                                               ComponentMask({true, true, true, false}));
                                               
    boundary_functions.clear();
    
    //Cylinder Boundary Conditions
    boundary_functions[6] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                               boundary_functions,
                                               boundary_values,
                                               ComponentMask({true, true, true, false}));
     boundary_functions.clear();

    // Up/Down Boundary Conditions
    boundary_functions[2] = &zero_function;
    boundary_functions[4] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             ComponentMask(
                                               {false, true, false, false})); 
    boundary_functions.clear();

    // Left/Right Boundary Conditions
    boundary_functions[3] = &zero_function;
    boundary_functions[5] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             ComponentMask(
                                               {false, false, true, false}));

    boundary_functions.clear();
    MatrixTools::apply_boundary_values(
    boundary_values, system_matrix, solution, system_rhs, false);
  }
}




void
NavierStokes::solve_time_step()
{
  pcout << "===============================================" << std::endl;

  SolverControl solver_control(10000, 1e-6 * system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  // TODO: implement case
  PreconditionBlockDiagonal preconditioner;
  preconditioner.initialize(system_matrix.block(0, 0),
                             pressure_mass.block(1, 1));

  //PreconditionIdentity preconditioner;

  // PreconditionBlockTriangular preconditioner;
  // preconditioner.initialize(system_matrix.block(0, 0),
  //                           pressure_mass.block(1, 1),
  //                           system_matrix.block(1, 0));

  pcout << "Solving the linear system" << std::endl;
  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " GMRES iterations"
        << std::endl;

  solution = solution_owned;
}

void NavierStokes::solve()
{
  pcout << "===============================================" << std::endl;
  time = 0.0;

  assemble_time_independent();

  u_0.set_time(time);
  VectorTools::interpolate(dof_handler, u_0, solution_owned);
  solution = solution_owned;

  // calculate coefficients
  calculate_coefficients();

  unsigned int time_step = 0;

  while (time < T)
  {
    time += deltat;
    ++time_step;

     pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":\n" << std::flush;

    assemble_system();
    solve_time_step();
    calculate_coefficients();
    
    output(time_step);
  }
  write_coefficients_on_files();
}


void NavierStokes::output(const unsigned int &time_step) const
{
  pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
  std::vector<std::string> names = {"velocity",
                                    "velocity",
                                    "velocity",
                                    "pressure"};

  data_out.add_data_vector(dof_handler,
                           solution,
                           names,
                           data_component_interpretation);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name = "output-stokes-3D";
  data_out.write_vtu_with_pvtu_record("./",
                                      output_file_name,
                                      time_step,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << std::endl;
  pcout << "===============================================" << std::endl;
}


void NavierStokes::calculate_coefficients()
{
  pcout << "===============================================" << std::endl;
  pcout << "Calcolo dei coefficienti" << std::endl;

  const unsigned int n_q_face = quadrature_face->size();

  FEFaceValues<dim> fe_values(*fe,
                                   *quadrature_face,
                                   update_values | update_gradients | update_normal_vectors |
                                   update_JxW_values);

  const double rho_nu = rho * nu;
  
  // Initialize variables to store drag and lift forces
  double drag_force = 0.0;
  double lift_force = 0.0;

  // Set up extractors for velocity and pressure components
  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pression(dim);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned() || !cell->at_boundary())
      continue;

    for (unsigned int q = 0; q < cell->n_faces(); ++q)
    {
      auto face = cell->face(q);
      if (!face->at_boundary() || face->boundary_id() != 6)
        continue;

      // Reinitialize face values for the current face
      fe_values.reinit(cell, q);

      std::vector<double> pression_value_loc(n_q_face);
      std::vector<Tensor<2, dim>> velocity_gradient_loc(n_q_face);
      fe_values[pression].get_function_values(solution, pression_value_loc);
      fe_values[velocity].get_function_gradients(solution, velocity_gradient_loc);

      // Assuming the normal vector is constant across the face for optimization
      Tensor<1, dim> normal_vector = -fe_values.normal_vector(0);

      for (unsigned int q = 0; q < n_q_face; ++q)
      {
        Tensor<2, dim> fluid_pressure;
        fluid_pressure[0][0] = pression_value_loc[q];
        fluid_pressure[1][1] = pression_value_loc[q];

        Tensor<1, dim> f = (rho_nu * velocity_gradient_loc[q] - fluid_pressure) * normal_vector * fe_values.JxW(q);
        drag_force += f[0];
        lift_force += f[1];
      }
    }
  }

  // Declare total variables to store the summed results
  double total_drag = 0.0, total_lift = 0.0;

  // Use MPI_Allreduce to sum the drag and lift forces across all processes, storing the result in all processes
  MPI_Allreduce(&drag_force, &total_drag, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&lift_force, &total_lift, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Every process now has the total drag and lift forces and can compute the coefficients
  drag_coefficients.push_back(constant_coeff * total_drag);
  lift_coefficients.push_back(constant_coeff * total_lift);
}

void NavierStokes::write_coefficients_on_files()
{
  const unsigned int current_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  pcout << "===============================================" << std::endl;
  pcout << "Writing coefficients on file" << std::endl;

  if (current_rank == 0) // Only the master process writes to the files
  {
    std::ofstream drag_file("drag_coefficient.csv"), lift_file("lift_coefficient.csv");

    if (drag_file && lift_file) // Check if files are successfully opened
    {
      // Write headers to both files
      drag_file << "Time,DragCoefficient" << std::endl;
      lift_file << "Time,LiftCoefficient" << std::endl;

      // Write coefficients to files
      for (unsigned int idx = 0; idx < drag_coefficients.size(); ++idx)
      {
        double time_point = idx * deltat; // Calculate the time point
        drag_file << time_point << "," << drag_coefficients[idx] << std::endl;
        lift_file << time_point << "," << lift_coefficients[idx] << std::endl;
      }

      pcout << "Drag and lift coefficients written to CSV files." << std::endl;
    }
    else
    {
      pcout << "Error opening output files!" << std::endl;
    }
  }

  pcout << "===============================================" << std::endl;
}
