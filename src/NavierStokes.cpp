#include "NavierStokes.hpp"

template<int dim>
void
NavierStokes<dim>::setup()
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

    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

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

    // velocity mass term.
    for (unsigned int c = 0; c < dim + 1; ++c) {
      for (unsigned int d = 0; d < dim + 1; ++d) {
        if (c == dim || d == dim)  // terms with pressure
          coupling[c][d] = DoFTools::none;
        else  // terms with no pressure
          coupling[c][d] = DoFTools::always;
      }
    }

    TrilinosWrappers::BlockSparsityPattern velocity_mass_sparsity(
        block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling,
                                    velocity_mass_sparsity);
    velocity_mass_sparsity.compress();

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
    constant_matrix.reinit(sparsity);
    velocity_mass.reinit(velocity_mass_sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);

    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  }
}

template<int dim>
void
NavierStokes<dim>::assemble_constant_matrices()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling time independent component" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  pcout<<"Velocity: "<<inlet_velocity.vel<<". Velocity case type: "<<inlet_velocity.case_type<< ". Using the skew-symmetric non-linear term: " <<use_skew<< std::endl;
  FEValues<dim>     fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_lhs_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> velocity_mass_cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  constant_matrix = 0.0;
  pressure_mass = 0.0;
  velocity_mass = 0.0;

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_lhs_matrix = 0.0;
      cell_pressure_mass_matrix = 0.0;
      velocity_mass_cell_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  
                  // Viscosity term A * Theta
                  cell_lhs_matrix(i, j) +=
                    nu * theta *
                    scalar_product(fe_values[velocity].gradient(i, q),
                                   fe_values[velocity].gradient(j, q)) *
                    fe_values.JxW(q);

                  // Mass Matrix M/deltat
                  cell_lhs_matrix(i, j) += scalar_product(fe_values[velocity].value(i, q),
                                                          fe_values[velocity].value(j, q)) 
                                                          / deltat * fe_values.JxW(q);
                    
                  // Pressure term in the momentum equation
                  cell_lhs_matrix(i, j) -= fe_values[velocity].divergence(i, q) *
                                       fe_values[pressure].value(j, q) *
                                       fe_values.JxW(q);

                  // Pressure term in the continuity equation
                  cell_lhs_matrix(i, j) -= fe_values[velocity].divergence(j, q) *
                                       fe_values[pressure].value(i, q) *
                                       fe_values.JxW(q);

                  // Pressure mass matrix.
                  cell_pressure_mass_matrix(i, j) +=
                    fe_values[pressure].value(i, q) *
                    fe_values[pressure].value(j, q) / nu * fe_values.JxW(q);
                  
                  // Matrix used for preconditioner
                  velocity_mass_cell_matrix(i, j) += scalar_product(fe_values[velocity].value(i, q),
                                                          fe_values[velocity].value(j, q)) 
                                                          / deltat * fe_values.JxW(q);                     
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      constant_matrix.add(dof_indices, cell_lhs_matrix);
      pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
      velocity_mass.add(dof_indices, velocity_mass_cell_matrix);
    }

  constant_matrix.compress(VectorOperation::add);
  pressure_mass.compress(VectorOperation::add);
  velocity_mass.compress(VectorOperation::add);
}


template <int dim>
void NavierStokes<dim>::assemble(const double &time)
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(*fe,
                                       *quadrature_face,
                                       update_values | update_quadrature_points |
                                           update_normal_vectors |
                                           update_JxW_values);

  FullMatrix<double> non_linear_contribution(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix.copy_from(constant_matrix);
  system_rhs = 0.0;

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<1, dim>> velocity_loc(n_q);
  std::vector<Tensor<2,dim>> velocity_gradient_loc(n_q);
  

  unsigned int neumann_boundary_id;
  if constexpr(dim==2) neumann_boundary_id=3;
  if constexpr(dim==3) neumann_boundary_id=1;
  
	for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    non_linear_contribution = 0.0;
    cell_rhs = 0.0;

    fe_values[velocity].get_function_values(solution, velocity_loc);
    fe_values[velocity].get_function_gradients(solution, velocity_gradient_loc);
  
  Tensor<1, dim> nonlinear_term;
  Tensor<1, dim> transpose_term;
    for (unsigned int q = 0; q < n_q; ++q)
    {

      Vector<double> forcing_term_loc(dim);
      forcing_term.vector_value(fe_values.quadrature_point(q),
                                forcing_term_loc);
      Tensor<1, dim> forcing_term_tensor;
      for (unsigned int d = 0; d < dim; ++d)
        forcing_term_tensor[d] = forcing_term_loc[d];

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {

        if (use_skew){ // Skew symmetric representation of non linear term
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            { 
              // Reset the nonlinear and transpose terms.
              nonlinear_term = 0.0;
              transpose_term = 0.0;
              for (unsigned int k = 0; k < dim; k++) {
                for (unsigned int l = 0; l < dim; l++) {
                  // Calculate (u . nabla) u.
                  nonlinear_term[k] += velocity_loc[q][l] *
                                        fe_values[velocity].gradient(j, q)[k][l];
                  // Calculate (nabla u)^T u.
                  transpose_term[k] += velocity_loc[q][k] *
                                        fe_values[velocity].gradient(j, q)[l][k];
                }
              }
              // Add the skew-symmetric term (u . nabla) u + (nabla u)^T u to the matrix.
              non_linear_contribution(i, j) +=
                  0.5 * (scalar_product(nonlinear_term, fe_values[velocity].value(i, q)) +
                          scalar_product(transpose_term, fe_values[velocity].value(i, q))) *
                  fe_values.JxW(q);
            }
        }
        else{
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Semi-implicit treatment of the non linear term
                  non_linear_contribution(i, j) += velocity_loc[q] * fe_values[velocity].gradient(j, q) *
                                        fe_values[velocity].value(i, q) *
                                        fe_values.JxW(q);
                }
        }
      
        // Forcing term
        cell_rhs(i) += scalar_product(forcing_term_tensor,
                                      fe_values[velocity].value(i, q)) *
                      fe_values.JxW(q);
        cell_rhs(i) += scalar_product(velocity_loc[q],
                                      fe_values[velocity].value(i, q)) /
                      deltat * fe_values.JxW(q);
      }
    }

    // Neumann BCs
    if (cell->at_boundary())
      {
        for (unsigned int f = 0; f < cell->n_faces(); ++f)
          {
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == neumann_boundary_id)
              {
                fe_face_values.reinit(cell, f);

                for (unsigned int q = 0; q < n_q; ++q)
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

    system_matrix.add(dof_indices, non_linear_contribution);
    system_rhs.add(dof_indices, cell_rhs);
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  
  // Dirichlet BCs
  std::map<types::global_dof_index, double>           boundary_values;
  std::map<types::boundary_id, const Function<dim> *> boundary_functions;
  Functions::ZeroFunction<dim> zero_function(dim + 1);

  // Dirichlet boundary conditions for 3D case
  if constexpr(dim==3){
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
  
  // Dirichlet boundary conditions for 2D case
  else{

    ComponentMask mask({true,true,false});
    inlet_velocity.set_time(time);
    boundary_functions[1] = &inlet_velocity;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             mask);

    boundary_functions[0] = &zero_function;
    boundary_functions[2] = &zero_function;
    boundary_functions[4] = &zero_function;
    boundary_functions[5] = &zero_function;
    boundary_functions[6] = &zero_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values,
                                             mask);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, true);
  
  }
}


template<int dim>
void
NavierStokes<dim>::solve_time_step()
{
  pcout << "===============================================" << std::endl;

  SolverControl solver_control(20000, 1e-4 * system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  // Apply preconditioners
if (prec == 0) {
    PreconditionBlockDiagonal preconditioner;
    preconditioner.initialize(system_matrix.block(0, 0),
                              pressure_mass.block(1, 1));
    pcout << "Solving the linear system" << std::endl;

    Timer timer;
    timer.start();
    solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
    timer.stop();
    
    pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;
    pcout << "  Time taken: " << timer.wall_time() << " seconds" << std::endl;

    solution = solution_owned;
} else if (prec == 1) {
    pcout << "Preconditioner SIMPLE" << std::endl;
    PreconditionSIMPLE preconditioner;
    preconditioner.initialize(
            system_matrix.block(0, 0), system_matrix.block(1, 0),
            system_matrix.block(0, 1), solution_owned);
    pcout << "Solving the linear system" << std::endl;

    Timer timer;
    timer.start();
    solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
    timer.stop();
    
    pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;
    pcout << "  Time taken: " << timer.wall_time() << " seconds" << std::endl;

    solution = solution_owned;
} else if (prec == 2) {
    PreconditionaSIMPLE preconditioner;
    pcout << "Preconditioner aSIMPLE" << std::endl;
    preconditioner.initialize(
            system_matrix.block(0, 0), system_matrix.block(1, 0),
            system_matrix.block(0, 1), solution_owned);
    pcout << "Solving the linear system" << std::endl;

    Timer timer;
    timer.start();
    solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
    timer.stop();
    
    pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;
    pcout << "  Time taken: " << timer.wall_time() << " seconds" << std::endl;

    solution = solution_owned;
} else if (prec == 3) {
    pcout << "Preconditioner Yosida" << std::endl;
    PreconditionYosida preconditioner;
    preconditioner.initialize(system_matrix.block(0, 0), system_matrix.block(1, 0),
            system_matrix.block(0, 1), velocity_mass.block(0, 0), solution_owned);
    pcout << "Solving the linear system" << std::endl;

    Timer timer;
    timer.start();
    solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
    timer.stop();
    
    pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;
    pcout << "  Time taken: " << timer.wall_time() << " seconds" << std::endl;

    solution = solution_owned;
}
}

template<int dim>
void NavierStokes<dim>::solve()
{
  pcout << "===============================================" << std::endl;
  time = 0.0;

  // Assemble constant matrices
  assemble_constant_matrices();

  u_0.set_time(time);
  VectorTools::interpolate(dof_handler, u_0, solution_owned);
  solution = solution_owned;

  unsigned int current_time_step = 0;

  while (time < T)
  {
    time += deltat;
    ++current_time_step;

     pcout << "n = " << std::setw(3) << current_time_step << ", t = " << std::setw(5)
            << time << ":\n" << std::flush;

    assemble(time);
    solve_time_step();
    calculate_coefficients(time);
    
    output(current_time_step);
  }
  write_coefficients_on_files();
}


template<int dim>
void NavierStokes<dim>::output(const unsigned int &time_step) const
{
  pcout << "===============================================" << std::endl;

  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

  if constexpr(dim == 2)
  	data_out.add_data_vector(dof_handler,
  	                         solution,
  	                         {"velocity","velocity","pressure"},
  	                         data_component_interpretation);
	else if constexpr(dim == 3)
  	data_out.add_data_vector(dof_handler,
  	                         solution,
 		                         {"velocity","velocity","velocity","pressure"},
  	                         data_component_interpretation);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::string output_file_name;
	if constexpr(dim == 2)
  	output_file_name = "output-navier-stokes-2D";
	else if constexpr(dim == 3)
  	output_file_name = "output-navier-stokes-3D";

  data_out.write_vtu_with_pvtu_record("./",
                                      output_file_name,
                                      time_step,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << std::endl;
  pcout << "===============================================" << std::endl;
}


template<int dim>
void NavierStokes<dim>::calculate_coefficients(double t)
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

  double Re = get_reynolds_number(t);

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
  double total_drag_force = 0.0, total_lift_force = 0.0;

  // Use MPI_Allreduce to sum the drag and lift forces across all processes, storing the result in all processes
  MPI_Allreduce(&drag_force, &total_drag_force, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&lift_force, &total_lift_force, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Every process now has the total drag and lift forces and can compute the coefficients
  
  drag_coefficients.push_back(get_drag_lift_multiplicative_const(t) * total_drag_force);
  lift_coefficients.push_back(get_drag_lift_multiplicative_const(t) * total_lift_force);
  
  pcout << "Coefficints at time " << t << std::endl;
  pcout << "Drag: " << drag_coefficients.back() << std::endl;
  pcout << "Lift: " << lift_coefficients.back()<< std::endl;
  pcout << "Re: " << Re << std::endl;
}

template<int dim>
void NavierStokes<dim>::write_coefficients_on_files()
{
  const unsigned int current_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  pcout << "===============================================" << std::endl;
  pcout << "Writing coefficients on file" << std::endl;
  if (current_rank == 0) 
  {

    std::ofstream drag_file("drag_coefficient.csv"), lift_file("lift_coefficient.csv");

    if (drag_file && lift_file)
    {
      // Write headers
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