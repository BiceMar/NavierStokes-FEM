#ifndef STOKES_HPP
#define STOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/timer.h>

#include <fstream>
#include <iostream>
#include <memory>

using namespace dealii;

template<int dim>
class NavierStokes
{
public:
  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override
    {
      for (unsigned int i = 0; i < dim - 1; ++i)
        values[i] = 0.0;

      values[dim - 1] = -g;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      if (component == dim - 1)
        return -g;
      else
        return 0.0;
    }

  protected:
    const double g = 0.0;
  };

  class InletVelocity : public Function<dim>
  {
  public:
      InletVelocity(int case_type, double vel)   // Default to case 1 if not specified
          : Function<dim>(dim + 1), vel(vel), case_type(case_type) // Inizializza vel prima di case_type
      {}

      double mean_velocity_value(double t) const {
          double base_velocity;

          // Handle 2D flow conditions
          if constexpr(dim==2){
              // Calculate the squared velocity profile based on the cylinder height and normalize by H squared.
              // This simplifies the cylinder influence calculation as a portion of the height squared.
              base_velocity = 4.0 * vel * std::pow(cylinder_height / 2.0, 2) / std::pow(cylinder_height, 2);
              if (case_type == 1) return base_velocity * 2.0/3.0;
              if (case_type == 2) return base_velocity * std::sin(M_PI * t / 8.0);
          }
          // Handle 3D flow conditions
          else if constexpr(dim==3){
              // Calculate the fourth power of velocity profile based on the cylinder height and normalize by H to the fourth power.
              // This assumes an even greater influence of the height due to additional dimension considerations.
              base_velocity = 16.0 * vel * std::pow(cylinder_height / 2.0, 4) / std::pow(cylinder_height, 4);
              if (case_type == 1) return base_velocity * 4.0/9.0;
              if (case_type == 2) return base_velocity * std::sin(M_PI * t / 8.0);
          }
          return 0.0; 
      }

      virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override {

          if constexpr(dim == 2){
            if (case_type == 1) { // 2D steady
              values[0] = 4.0 * vel * p[1] * (cylinder_height - p[1]) / std::pow(cylinder_height, 2);
            }
            if (case_type == 2){ // 2D unsteady
              values[0] = 4.0 * vel * p[1] * (cylinder_height - p[1]) * std::sin(M_PI * this->get_time() / 8.0) /  std::pow(cylinder_height, 2);
            }
          }
          else if constexpr(dim == 3){
              if (case_type == 1) { // 3D steady
                values[0] = 16.0 * vel * p[1] * p[2] *(cylinder_height - p[1]) * (cylinder_height - p[2]) / std::pow(cylinder_height, 4);
              } 
              if (case_type == 2) { // 3D unsteady
                values[0] = 16.0 * vel * p[1] * p[2] *(cylinder_height - p[1]) * (cylinder_height - p[2]) * std::sin(M_PI * this->get_time() / 8.0) / std::pow(cylinder_height, 4);
              }
          }  
          for (unsigned int i = 1; i < dim + 1; ++i)
              values[i] = 0.0;
      }

      virtual double value(const Point<dim> &p, const unsigned int component = 0) const override {
          if (component == 0) {
            if constexpr(dim == 2){
              if (case_type == 1) return 4.0 * vel * p[1] * (cylinder_height - p[1]) / std::pow(cylinder_height, 2);
              if (case_type == 2) return 4.0 * vel * p[1] * (cylinder_height - p[1]) * std::sin(M_PI * this->get_time() / 8.0) /  std::pow(cylinder_height, 2);
            }
            if constexpr(dim == 3){
              if (case_type == 1) return 16.0 * vel * p[1] * p[2] *(cylinder_height - p[1]) * (cylinder_height - p[2]) / std::pow(cylinder_height, 4);     
              if (case_type == 2) return 16.0 * vel * p[1] * p[2] *(cylinder_height- p[1]) * (cylinder_height- p[2]) * std::sin(M_PI * this->get_time() / 8.0) / std::pow(cylinder_height, 4);
            }
          }             
            return 0.0;
    
      }
    
      double vel; 
      int case_type; 

     
  };

  // Neumann BCs
  class FunctionH : public Function<dim>
  {
  public:
    // Constructor.
    FunctionH()
    {}

    virtual double
    value(const Point<dim> & /*p*/, const unsigned int /*component*/) const override
    {
      return 0.;
    }
  };

  
  // Dirichlet BCs
  class FunctionG : public Function<dim>
  {
  public:
    // Constructor.
    FunctionG() : Function<dim>(dim + 1)
    {
    }

    virtual void
    vector_value(const Point<dim> & /*p*/, Vector<double> &values) const override
    {
      for (unsigned int i = 0; i < dim + 1; ++i)
        values[i] = 0.0;
    }

    virtual double
    value(const Point<dim> & /*p*/, const unsigned int /*component*/) const override
    {
      return 0.;
    }
  };


  // Function for the initial condition.
  class FunctionU0 : public Function<dim>
  {  
  public:
    virtual void
    vector_value(const Point<dim> & /*p*/,
                 Vector<double> &values) const override
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
    }

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Since we're working with block matrices, we need to make our own
  // preconditioner class. A preconditioner class can be any class that exposes
  // a vmult method that applies the inverse of the preconditioner.

  // Identity preconditioner.
  class PreconditionIdentity
  {
  public:
    // Application of the preconditioner: we just copy the input vector (src)
    // into the output vector (dst).
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      dst = src;
    }

  protected:
  };

  // Block-diagonal preconditioner.
  class PreconditionBlockDiagonal
  {
  public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      SolverControl                           solver_control_velocity(10000,
                                            1e-2 * src.block(0).l2_norm()); // 
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
        solver_control_velocity);
      solver_cg_velocity.solve(*velocity_stiffness,
                               dst.block(0), // 
                               src.block(0),
                               preconditioner_velocity);

      SolverControl                           solver_control_pressure(10000,
                                            1e-2 * src.block(1).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
        solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               src.block(1),
                               preconditioner_pressure);
    }

  protected:
    // Velocity stiffness matrix.
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    // Pressure mass matrix.
    const TrilinosWrappers::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionILU preconditioner_pressure;
  };

  class PreconditionYosida {
  public:
   // Initialize the preconditioner.
   void initialize(const TrilinosWrappers::SparseMatrix &F_,
                   const TrilinosWrappers::SparseMatrix &B_,
                   const TrilinosWrappers::SparseMatrix &Bt_,
                   const TrilinosWrappers::SparseMatrix &M_dt_,
                   const TrilinosWrappers::MPI::BlockVector &vec)
      {
        // Save a reference to the input matrices.
        F = &F_;
        B  = &B_;
        B_trans = &Bt_;

        // Save the inverse diagonal of M_dt.

        diag_D_inv.reinit(vec.block(0));
        for (unsigned int index : diag_D_inv.locally_owned_elements()) {
          diag_D_inv[index] = 1.0 / M_dt_.diag_element(index);
        }

        // Create the matrix S.
        B->mmult(S, *B_trans, diag_D_inv);

        // Initialization for the preconditioners.
        preconditioner_F.initialize(*F);
        preconditioner_S.initialize(S);

      }
  
   // Application of the preconditioner.
   void vmult(TrilinosWrappers::MPI::BlockVector &dst,
              const TrilinosWrappers::MPI::BlockVector &src) const 
      {
        const unsigned int maxiter = 10000;
        const double tol = 1e-2;
        inter_sol.reinit(src);
        
        // Solve [F 0; B -S]sol1 = src.
        
        // First solve F*sol1_u = src_u.
        inter_sol.block(0) = dst.block(0);
        SolverControl solver_control_F(maxiter, tol * src.block(0).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
        solver_F.solve(*F, inter_sol.block(0), src.block(0), preconditioner_F);
        
        //Solve S*sol1_p = B*sol1_u - src_p.
        inter_sol.block(1) = src.block(1);
        B->vmult_add(inter_sol.block(1), inter_sol.block(0));
        SolverControl solver_control_S(maxiter, tol * inter_sol.block(1).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
        solver_S.solve(S, dst.block(1), inter_sol.block(1), preconditioner_S);

        // Solve [I F^-1*B^T; 0 I]dst = sol1.
        inter_sol2 = src.block(0);
        dst.block(0) = inter_sol.block(0);
        B_trans->vmult(inter_sol.block(0), dst.block(1));
        SolverControl solver_control_F2(maxiter, tol* inter_sol.block(0).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F2(solver_control_F2);
        solver_F2.solve(*F, inter_sol2, inter_sol.block(0), preconditioner_F);
        dst.block(0) -= inter_sol2;
      }
  
  protected:
   
   const TrilinosWrappers::SparseMatrix *F;
   const TrilinosWrappers::SparseMatrix *B;
   const TrilinosWrappers::SparseMatrix *B_trans;
  
   // Matrix and vector for intermediate computations
   TrilinosWrappers::SparseMatrix S;
   TrilinosWrappers::SparseMatrix neg_B;
   mutable TrilinosWrappers::MPI::BlockVector inter_sol;
   mutable TrilinosWrappers::MPI::Vector inter_sol2;
   TrilinosWrappers::MPI::Vector diag_D_inv;

  // ILU preconditioners for F and S_tilde
   TrilinosWrappers::PreconditionILU preconditioner_F;
   TrilinosWrappers::PreconditionILU preconditioner_S;
  };

   // SIMPLE preconditioner.
  class PreconditionSIMPLE 
  {

  public:
    void
    initialize(const TrilinosWrappers::SparseMatrix &F_,
               const TrilinosWrappers::SparseMatrix &B_,
               const TrilinosWrappers::SparseMatrix &B_t,
               const TrilinosWrappers::MPI::BlockVector &sol_owned)
    {
      // Save the references to the input matrices.
      F = &F_;
      B = &B_;
      B_trans = &B_t;

      // Create the inverse of the diagonal of F
      diag_D_inv.reinit(sol_owned.block(0));
      for (unsigned int i : diag_D_inv.locally_owned_elements())
      {
        double temp = F->diag_element(i);
        diag_D_inv[i] = 1.0 / temp;
      }

      // Create Schur complement approximation matrix S_tilde
      B_.mmult(S_tilde, B_t, diag_D_inv);

      // Initialize the preconditioners for F and S_tilde
      preconditioner_F.initialize(*F);
      preconditioner_S.initialize(S_tilde);
    }

    void
    vmult(TrilinosWrappers::MPI::BlockVector &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      const unsigned int maxiter = 10000;
      const double tol = 1e-2;
      SolverControl solver_F(maxiter, tol * src.block(0).l2_norm());

      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres(solver_F);

      TrilinosWrappers::MPI::Vector sol_u = src.block(0);
      TrilinosWrappers::MPI::Vector sol_p = src.block(1);
      TrilinosWrappers::MPI::Vector inter_sol = src.block(1);

      //Solving F * sol_u = src_u
      solver_gmres.solve(*F, sol_u, src.block(0), preconditioner_F);

      // Compute the residual inter_sol = B * sol_u - src_p
      B->vmult(inter_sol, sol_u);
      inter_sol -= src.block(1);

      //Solving S_tilde * sol_p = iter_sol
      SolverControl solver_S(maxiter, tol * inter_sol.l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_S(solver_S);
      solver_gmres_S.solve(S_tilde, sol_p, inter_sol, preconditioner_S);

      // Update destination vector 
      dst.block(1) = sol_p;
      dst.block(1) *= 1. / alpha;

      // Compute the residual dst_u = sol_u - D^-1 * B^T * sol_p
      B_trans->vmult(dst.block(0), dst.block(1));
  
      dst.block(0).scale(diag_D_inv);
      dst.block(0) -= sol_u;
      dst.block(0) *= -1.;
    }

  protected:
    const double alpha = 0.5;

    const TrilinosWrappers::SparseMatrix *F;
    const TrilinosWrappers::SparseMatrix *B_trans;
    const TrilinosWrappers::SparseMatrix *B;
     // Matrix and vector for intermediate computations
    TrilinosWrappers::SparseMatrix S_tilde;
    TrilinosWrappers::MPI::Vector diag_D_inv;
    // ILU preconditioners for F and S_tilde
    TrilinosWrappers::PreconditionILU preconditioner_F;
    TrilinosWrappers::PreconditionILU preconditioner_S;
  };

  // aSIMPLE preconditioner
  class PreconditionaSIMPLE
  {
  public:
    // Initialization function for setting up matrices and vectors
    void initialize(const TrilinosWrappers::SparseMatrix &F_,
                    const TrilinosWrappers::SparseMatrix &B_,
                    const TrilinosWrappers::SparseMatrix &B_t,
                    const TrilinosWrappers::MPI::BlockVector &sol_owned)
    {
      // Assign input matrices to class members
      F = &F_;
      B = &B_;
      B_T = &B_t;

      // Resize and initialize diagonal vectors based on input vector block
      D_inv.reinit(sol_owned.block(0));
      neg_D.reinit(sol_owned.block(0));

      // Compute inverse of diagonal matrix
      for (unsigned int i : neg_D.locally_owned_elements())
      {
        neg_D[i] = -(F->diag_element(i));            
        D_inv[i] = 1.0 / F->diag_element(i);
      }

      // Compute Schur complement matrix S = B * D_inv * B_T
      B->mmult(S, *B_T, D_inv);

      // Initialize ILU preconditioners for matrices F and S
      preconditionerF.initialize(*F);
      preconditionerS.initialize(S);
    }

    void vmult(TrilinosWrappers::MPI::BlockVector &dst,
              const TrilinosWrappers::MPI::BlockVector &src) const
    {
      const unsigned int maxiter = 10000;
      const double tol = 1e-2;

      SolverControl solver_F(maxiter, tol * src.block(0).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_F(solver_F);

      tmp.reinit(src.block(1));

      // Solve F * dst_u = src_u
      solver_gmres_F.solve(*F, dst.block(0), src.block(0), preconditionerF);
      dst.block(1) = src.block(1);
      
      // Compute the residual tmp = B * dst_u - src_p
      B->vmult(dst.block(1), dst.block(0));
      dst.block(1).sadd(-1.0, src.block(1));
      tmp = dst.block(1);
      
      // Solve the linear system S * dst_p = tmp 
      SolverControl solver_S(maxiter, tol * tmp.l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_S(solver_S);
      solver_gmres_S.solve(S, dst.block(1), tmp, preconditionerS);
      
      dst.block(0).scale(neg_D);
      dst.block(1) *= 1.0 / alpha;

      B_T->vmult_add(dst.block(0), dst.block(1));

      
      dst.block(0).scale(D_inv);
    }

  protected:

    const TrilinosWrappers::SparseMatrix *F;
    const TrilinosWrappers::SparseMatrix *B_T;
    const TrilinosWrappers::SparseMatrix *B;

    // Schur complement matrix
    TrilinosWrappers::SparseMatrix S;

    // ILU preconditioners for matrices F and S
    TrilinosWrappers::PreconditionILU preconditionerF;
    TrilinosWrappers::PreconditionILU preconditionerS;

    // Diagonal and inverse diagonal vectors
    TrilinosWrappers::MPI::Vector neg_D;
    TrilinosWrappers::MPI::Vector D_inv;

    // Temporary vectors for intermediate computations
    mutable TrilinosWrappers::MPI::Vector tmp;
    mutable TrilinosWrappers::MPI::Vector tmp2;

    // Constant scalar alpha for scaling
    const double alpha = 0.5;
  };

  NavierStokes(const std::string &mesh_file_name_,
                           const unsigned int &degree_velocity_,
                           const unsigned int &degree_pressure_,
                           const double &_T,
                           const double &deltat_,
                           const double &theta_,
                           const double nu_,
                           const double p_out_,
                           const double rho_,
                           const int case_type_,
                           const double vel_,
                           const unsigned int prec_,
                           const unsigned int use_skew_)
  : mesh_file_name(mesh_file_name_),
    degree_velocity(degree_velocity_),
    degree_pressure(degree_pressure_),
    T(_T),
    deltat(deltat_),
    theta(theta_),
    nu(nu_),
    p_out(p_out_),
    rho(rho_),
    inlet_velocity(case_type_, vel_),
    prec(prec_),
    use_skew(use_skew_),
    mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
    mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
    pcout(std::cout, mpi_rank == 0),
    mesh(MPI_COMM_WORLD)
    {
    }


  void
  setup();

  void
  solve();

protected:

  // Assemble constant matrices
  void
  assemble_constant_matrices();

  // Assemble system 
  void assemble(const double &time);

  // Solve system
  void
  solve_time_step();

  // Output results
  void
  output(const unsigned int &time_step) const;

  // Calculate coefficient
  void
  calculate_coefficients(double t);

  // Write coefficients on file
  void
  write_coefficients_on_files();
  
  // Mesh file
  const std::string mesh_file_name;
 
  // Polynomial degree
  const unsigned int degree_velocity;
  const unsigned int degree_pressure;

  // Total Time 
  double T;

  // Time step
  double deltat;

  // Theta parameter of the theta method.
  double theta;

  // Kinematic viscosity [m2/s]
  double nu = 0.001;   

  // Outlet pressure [Pa]
  double p_out = 10.0;

  // density [Kg/m^3]
  double rho = 1.0; 

  // Inlet velocity
  InletVelocity inlet_velocity;

  int prec = 0; // 0->diagonal, 1->SIMPLE, 2->ASIMPLE, 2->yosida

  int use_skew = 0; 

  // MPI processes
  const unsigned int mpi_size;

  // Current MPI process
  const unsigned int mpi_rank;

  // Parallel output stream
  ConditionalOStream pcout;

//----------------------------------------------------------------------------
  
  // Cylinder diameter, fixed parameter
  static constexpr double cylinder_diameter = 0.1;

  // Cylinder height, fixed parameter
  static constexpr double cylinder_height = 0.41;

  // Forcing term
  ForcingTerm forcing_term;

  // Time
  double time;

  // Initial Condition
  FunctionU0 u_0;

  // function g
  FunctionG function_g;

  // function h
  FunctionH function_h;

  // Vector of all the drag coefficients
  std::vector<double> drag_coefficients; 
  
  // Vector of all the drag coefficients
  std::vector<double> lift_coefficients;

  // Retrive Drag/Lift coefficient multiplicative constant
  double get_drag_lift_multiplicative_const(double t) {
    double multiplicative_const;

    if constexpr(dim == 2) multiplicative_const = 2.0 / (rho * inlet_velocity.mean_velocity_value(t) * inlet_velocity.mean_velocity_value(t) * cylinder_height); 
    if constexpr(dim == 3) multiplicative_const = 2.0 / (rho * inlet_velocity.mean_velocity_value(t) * inlet_velocity.mean_velocity_value(t) * cylinder_diameter * cylinder_height); 
    
    return multiplicative_const; 
  }

  // Retrive Reynolds number
  double get_reynolds_number(double t) {
    return inlet_velocity.mean_velocity_value(t) * cylinder_diameter / nu;
  }

//----------------------------------------------------------------------------

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for face integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs owned by current process in the velocity and pressure blocks.
  std::vector<IndexSet> block_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // DoFs relevant to current process in the velocity and pressure blocks.
  std::vector<IndexSet> block_relevant_dofs;

  // cosntant matrix
  TrilinosWrappers::BlockSparseMatrix constant_matrix;

  // (M / deltat - (1 - theta) * A)
  TrilinosWrappers::BlockSparseMatrix rhs_matrix;


  // Pressure mass matrix, needed for preconditioning. We use a block matrix for
  // convenience, but in practice we only look at the pressure-pressure block.
  TrilinosWrappers::BlockSparseMatrix pressure_mass;
  TrilinosWrappers::BlockSparseMatrix velocity_mass;

  // System matrix.
  TrilinosWrappers::BlockSparseMatrix system_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::BlockVector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::BlockVector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::BlockVector solution;
};

#endif