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

#include <fstream>
#include <iostream>
#include <memory>

using namespace dealii;

// Class implementing a solver for the Stokes problem.
class NavierStokes
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

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

  // Function for inlet velocity. This actually returns an object with four
  // components (one for each velocity component, and one for the pressure), but
  // then only the first three are really used (see the component mask when
  // applying boundary conditions at the end of assembly). If we only return
  // three components, however, we may get an error message due to this function
  // being incompatible with the finite element space.
  class InletVelocity : public Function<dim>
{
public:
    InletVelocity(int case_type = 1, double vel = 2.25)   // Default to case 1 if not specified
        : Function<dim>(dim + 1), vel(vel), case_type(case_type) // Inizializza vel prima di case_type
    {}

    double mean_value() const {
        return vel; // [m/s]
    }
    double maxVelocity() const {
        return 16 * vel;
    }

    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override {
        if (case_type == 1) {
            values[0] = 16.0 * vel * p[1] * p[2] *(H - p[1]) * (H - p[2]) / std::pow(H, 4);
        } else {
            values[0] = 16 * vel * p[1] * p[2] *(H - p[1]) * (H - p[2]) * std::sin(M_PI * get_time() / 8.0) / std::pow(H, 4);
        }
        
        for (unsigned int i = 1; i < dim + 1; ++i)
            values[i] = 0.0;
    }

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override {
        if (component == 0) {
            if (case_type == 1) {
                return 16.0 * vel * p[1] * p[2] *(H - p[1]) * (H - p[2]) / std::pow(H, 4);
            } else {
                return 16 * vel * p[1] * p[2] *(H - p[1]) * (H - p[2]) * std::sin(M_PI * get_time() / 8.0) / std::pow(H, 4);
            }
        } else {
            return 0.0;
        }
    }

public:
    double vel = 0.45; // [m/s] prev val: 2.25
    double H = 0.41;
    int case_type = 0; // Attributo per selezionare il caso per il calcolo
};


  // Function for the initial condition.
  class FunctionU0 : public Function<dim>
  {
  //public:
  //  virtual double
  //  value(const Point<dim> &p,
  //        const unsigned int /*component*/ = 0) const override
  //  {
  //    return p[0] * (1.0 - p[0]) * p[1] * (1.0 - p[1]) * p[2] * (1.0 - p[2]);
  //  }
  
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

  // Block-triangular preconditioner.
  // class PreconditionBlockTriangular
  // {
  // public:
  //   // Initialize the preconditioner, given the velocity stiffness matrix, the
  //   // pressure mass matrix.
  //   void
  //   initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
  //              const TrilinosWrappers::SparseMatrix &pressure_mass_,
  //              const TrilinosWrappers::SparseMatrix &B_)
  //   {
  //     velocity_stiffness = &velocity_stiffness_;
  //     pressure_mass      = &pressure_mass_;
  //     B                  = &B_;
  //     preconditioner_velocity.initialize(velocity_stiffness_);
  //     preconditioner_pressure.initialize(pressure_mass_);
  //   }

  //   // Application of the preconditioner.
  //   void
  //   vmult(TrilinosWrappers::MPI::BlockVector       &dst,
  //         const TrilinosWrappers::MPI::BlockVector &src) const
  //   {
  //     SolverControl                           solver_control_velocity(1000,
  //                                           1e-2 * src.block(0).l2_norm());
  //     SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
  //       solver_control_velocity);
  //     solver_cg_velocity.solve(*velocity_stiffness,
  //                              dst.block(0),
  //                              src.block(0),
  //                              preconditioner_velocity);

  //     tmp.reinit(src.block(1));
  //     B->vmult(tmp, dst.block(0));
  //     tmp.sadd(-1.0, src.block(1));

  //     SolverControl                           solver_control_pressure(1000,
  //                                           1e-2 * src.block(1).l2_norm());
  //     SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
  //       solver_control_pressure);
  //     solver_cg_pressure.solve(*pressure_mass,
  //                              dst.block(1),
  //                              tmp,
  //                              preconditioner_pressure);
  //   }

//   protected:
//     // Velocity stiffness matrix.
//     const TrilinosWrappers::SparseMatrix *velocity_stiffness;

//     // Preconditioner used for the velocity block.
//     TrilinosWrappers::PreconditionILU preconditioner_velocity;

//     // Pressure mass matrix.
//     //const TrilinosWrappers::SparseMatrix *pressure_mass;

//     // Preconditioner used for the pressure block.
//     TrilinosWrappers::PreconditionILU preconditioner_pressure;

//     // B matrix.
//     const TrilinosWrappers::SparseMatrix *B;

//     // Temporary vector.
//     mutable TrilinosWrappers::MPI::Vector tmp;
// };

  // class PreconditionIdentity
  // {
  // public:
  //   // Application of the preconditioner: we just copy the input vector (src)
  //   // into the output vector (dst).
  //   void
  //   vmult(TrilinosWrappers::MPI::BlockVector       &dst,
  //         const TrilinosWrappers::MPI::BlockVector &src) const
  //   {
  //     dst = src;
  //   }

  // protected:
  // };

  
  // PCD preconditioner.
  class PreconditionBlockPCD
  {
  public:
    // Initialize the preconditioner, given the F matrix, the
    // pressure mass matrix, the B matrix, the Ap matrix and
    // the Fp matrix.
    void
    initialize(const TrilinosWrappers::SparseMatrix &F_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_,
               const TrilinosWrappers::SparseMatrix &Bt_,
               const TrilinosWrappers::SparseMatrix &Ap_,
               const TrilinosWrappers::SparseMatrix &Fp_)
    {
      F                  = &F_;
      pressure_mass      = &pressure_mass_;
      Bt                 = &Bt_;
      Ap                 = &Ap_;
      Fp                 = &Fp_;

      preconditioner_F.initialize(F_);
      preconditioner_pressure.initialize(pressure_mass_);
      preconditioner_Ap.initialize(Ap_);  
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector &      dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {

      // block 0

      tmp1.reinit(src.block(1));
      tmp2.reinit(src.block(1));
      tmp3.reinit(src.block(0));

      SolverControl solver_control_pressure(1000, 1e-2 * src.block(0).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_pressure(solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               tmp1,
                               src.block(1),
                               preconditioner_pressure);
  
      Fp->vmult(tmp2, tmp1);

      SolverControl solver_control_Ap(1000, 1e-2 * src.block(0).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_Ap(solver_control_Ap);
      solver_cg_Ap.solve(*Ap,
                          tmp1,
                          tmp2,
                          preconditioner_Ap);

      Bt->vmult(tmp3, tmp1);

      tmp3.sadd(1.0, src.block(0));

      SolverControl solver_control_F(1000, 1e-2 * src.block(0).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_F(solver_control_F);
      solver_cg_F.solve (*F,
                          dst.block(0),
                          tmp3,
                          preconditioner_F);

      // block 1

      tmp1.reinit(src.block(1));
      tmp2.reinit(src.block(1));

      solver_cg_pressure.solve(*pressure_mass,
                               tmp1,
                               src.block(1),
                               preconditioner_pressure);
      
      Fp->vmult(tmp2, tmp1);

      solver_cg_Ap.solve(*Ap,
                          tmp1,
                          tmp2,
                          preconditioner_Ap);

      dst.block(1).sadd(-1.0, tmp1); 

    }

  protected:
    // F matrix.
    const TrilinosWrappers::SparseMatrix *F;

    // Preconditioner used for the F block.
    TrilinosWrappers::PreconditionILU preconditioner_F;

    // Pressure mass matrix.
    const TrilinosWrappers::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionILU preconditioner_pressure;

    // Bt matrix.
    const TrilinosWrappers::SparseMatrix *Bt;

    // Ap matrix.
    const TrilinosWrappers::SparseMatrix *Ap;

    // Preconditioner used for the Ap block.
    TrilinosWrappers::PreconditionILU preconditioner_Ap;

    // Fp matrix.
    const TrilinosWrappers::SparseMatrix *Fp;

    // Temporary vectors.
    mutable TrilinosWrappers::MPI::Vector tmp1;
    mutable TrilinosWrappers::MPI::Vector tmp2;
    mutable TrilinosWrappers::MPI::Vector tmp3;
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

      //Solving F * sol_u = src_u with GMRES
      solver_gmres.solve(*F, sol_u, src.block(0), preconditioner_F);

      // Compute the residual inter_sol = B * sol_u - src_p
      B->vmult(inter_sol, sol_u);
      inter_sol -= src.block(1);

      //Solving S_tilde * sol_p = iter_sol with GMRES
      SolverControl solver_S(maxiter, tol * inter_sol.l2_norm());
      //SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_S(solver_S);
      //solver_gmres_S.solve(S_tilde, sol_p, inter_sol, preconditioner_S);
      SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_S);
      solver_cg.solve(S_tilde, sol_p, inter_sol, preconditioner_S);

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

  class PreconditionYosida {
  public:
   // Initialize the preconditioner.
   void initialize(const TrilinosWrappers::SparseMatrix &F_matrix_,
                   const TrilinosWrappers::SparseMatrix &negB_matrix_,
                   const TrilinosWrappers::SparseMatrix &Bt_matrix_,
                   const TrilinosWrappers::SparseMatrix &M_dt_matrix_,
                   const TrilinosWrappers::MPI::BlockVector &vec,
                   const unsigned int &maxit_, const double &tol_)
      {
        maxit = maxit_;
        tol = tol_;
        // Save a reference to the input matrices.
        F_matrix = &F_matrix_;
        negB_matrix = &negB_matrix_;
        Bt_matrix = &Bt_matrix_;

        // Save the inverse diagonal of M_dt.
        Dinv_vector.reinit(vec.block(0));
        for (unsigned int index : Dinv_vector.locally_owned_elements()) {
          Dinv_vector[index] = 1.0 / M_dt_matrix_.diag_element(index);
        }

        // Create the matrix -S.
        negB_matrix->mmult(negS_matrix, *Bt_matrix, Dinv_vector);

        preconditioner_F->initialize(*F_matrix);
        preconditioner_S->initialize(negS_matrix);

      }
  
   // Application of the preconditioner.
   void vmult(TrilinosWrappers::MPI::BlockVector &dst,
              const TrilinosWrappers::MPI::BlockVector &src) const 
      {
        tmp.reinit(src);
        // Step 1: solve [F 0; B -S]sol1 = src.
        // Step 1.1: solve F*sol1_u = src_u.
        tmp.block(0) = dst.block(0);
        SolverControl solver_control_F(maxit, tol * src.block(0).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
        solver_F.solve(*F_matrix, tmp.block(0), src.block(0), *preconditioner_F);
        // Step 1.2: solve -S*sol1_p = -B*sol1_u + src_p.
        tmp.block(1) = src.block(1);
        negB_matrix->vmult_add(tmp.block(1), tmp.block(0));
        SolverControl solver_control_S(maxit, tol * tmp.block(1).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
        solver_S.solve(negS_matrix, dst.block(1), tmp.block(1), *preconditioner_S);

        // Step 2: solve [I F^-1*B^T; 0 I]dst = sol1.
        tmp_2 = src.block(0);
        dst.block(0) = tmp.block(0);
        Bt_matrix->vmult(tmp.block(0), dst.block(1));
        SolverControl solver_control_F2(maxit, tol * tmp.block(0).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_F2(solver_control_F);
        solver_gmres_F2.solve(*F_matrix, tmp_2, tmp.block(0), *preconditioner_F);
        dst.block(0) -= tmp_2;
      }
  
  private:
   // Matrix F (top left block of the system matrix).
   const TrilinosWrappers::SparseMatrix *F_matrix;
  
   // Matrix -B (bottom left block of the system matrix).
   const TrilinosWrappers::SparseMatrix *negB_matrix;
  
   // Matrix B^T (top right block of the system matrix).
   const TrilinosWrappers::SparseMatrix *Bt_matrix;
  
   // Matrix D^-1, inverse diagonal of M/deltat.
   TrilinosWrappers::MPI::Vector Dinv_vector;
  
   // Matrix -S := -B*D^-1*B^T.
   TrilinosWrappers::SparseMatrix negS_matrix;
  
   // Preconditioner used for the block multiplied by F.
   std::shared_ptr<TrilinosWrappers::PreconditionILU> preconditioner_F;
  
   // Preconditioner used for the block multiplied by S.
   std::shared_ptr<TrilinosWrappers::PreconditionILU> preconditioner_S;
  
   // Temporary vectors.
   mutable TrilinosWrappers::MPI::BlockVector tmp;
   mutable TrilinosWrappers::MPI::Vector tmp_2;
  
   // Maximum number of iterations for the inner solvers.
   unsigned int maxit;
  
   // Tolerance for the inner solvers.
   double tol;
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

    // Iterate over locally owned elements to initialize diagonal elements
    for (unsigned int i : neg_D.locally_owned_elements())
    {
      neg_D[i] = -(F->diag_element(i));                // Store negative of diagonal element
      D_inv[i] = 1.0 / F->diag_element(i);       // Store inverse of diagonal element
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
    // Define maximum iterations and tolerance for the solver
    const unsigned int maxiter = 10000;
    const double tol = 1e-2;

    // Setup GMRES solver for matrix F with specified tolerance
    SolverControl solver_F(maxiter, tol * src.block(0).l2_norm());
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres(solver_F);

    // Resize temporary vector based on second block of src
    tmp.reinit(src.block(1));

    // Solve F * dst.block(0) = src.block(0) using GMRES with preconditionerF
    solver_gmres.solve(*F, dst.block(0), src.block(0), preconditionerF);

    // Copy src.block(1) to dst.block(1)
    dst.block(1) = src.block(1);

    // Compute B * dst.block(0) and store result in dst.block(1)
    B->vmult(dst.block(1), dst.block(0));

    // Update dst.block(1) by subtracting src.block(1)
    dst.block(1).sadd(-1.0, src.block(1));

    // Copy updated dst.block(1) to temporary vector tmp
    tmp = dst.block(1);

    // Setup CG solver for matrix S with specified tolerance
    SolverControl solver_S(maxiter, tol * tmp.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> solver_cg(solver_S);

    // Solve S * dst.block(1) = tmp using CG with preconditionerS
    solver_cg.solve(S, dst.block(1), tmp, preconditionerS);

    // Scale dst.block(0) by neg_D
    dst.block(0).scale(neg_D);

    // Scale dst.block(1) by 1/alpha
    dst.block(1) *= 1.0 / alpha;

    // Compute B_T * dst.block(1) and add result to dst.block(0)
    B_T->vmult_add(dst.block(0), dst.block(1));

    // Scale dst.block(0) by D_inv
    dst.block(0).scale(D_inv);
  }

protected:
  // Pointers to sparse matrices
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

 // Setup system.
  void
  setup();

  NavierStokes(const std::string &mesh_file_name_,
                           const unsigned int &degree_velocity_,
                           const unsigned int &degree_pressure_,
                           const double &_T,
                           const double &deltat_,
                           const double &theta_,
                           double nu_,
                           double p_out_,
                           double rho_,
                           int case_type_,
                           double vel_,
                           int prec_)
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
    mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
    mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
    pcout(std::cout, mpi_rank == 0),
    mesh(MPI_COMM_WORLD)
    {
    }

  // Assemble system. We also assemble the pressure mass matrix (needed for the
  // preconditioner).
  void
  assemble_time_independent();

  // Assemble system for each time step.
  void assemble_system();

  // Solve system.
  void
  solve_time_step();

  void
  solve();

  // Output results.
  void
  output(const unsigned int &time_step) const;

  // Calculate coefficient.
  void
  calculate_coefficients();

  // Write coefficients on file.
  void
  write_coefficients_on_files();

protected:
  
  // Mesh file name.
  const std::string mesh_file_name;
 
  // Polynomial degree used for velocity.
  const unsigned int degree_velocity;

  // Polynomial degree used for pressure.
  const unsigned int degree_pressure;

  // Total Time 
  double T;

  // Time step.
  double deltat;

  // Theta parameter of the theta method.
  double theta;

  // Kinematic viscosity [m2/s].
  double nu = 0.001;   

  // Outlet pressure [Pa].
  double p_out = 10.0;

  // density [Kg/m^3].
  double rho = 1.0; 

  // Inlet velocity.
  InletVelocity inlet_velocity;

  int prec = 0; // 0->diagonal, 1->SIMPLE, 2->ASIMPLE

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

//----------------------------------------------------------------------------
  
  // Cylinder diameter [m].
  const double cylinder_diameter = 0.1;

  // Cylinder height [m].
  const double cylinder_height = 0.41;
  // Forcing term.
  ForcingTerm forcing_term;

  // Current Time
  double time;

  // Initial Condition
  FunctionU0 u_0;

  // Vector of all the drag coefficients.
  std::vector<double> drag_coefficients; 
  
  // Vector of all the drag coefficients.
  std::vector<double> lift_coefficients;

  // Drag/Lift coefficient multiplicative constant.
  const double constant_coeff_3D = 2.0 / (rho * inlet_velocity.mean_value() * inlet_velocity.mean_value() * cylinder_diameter * cylinder_height); 

  // Drag/Lift coefficient multiplicative constant.
  const double constant_coeff_2D = 2.0 / (rho * inlet_velocity.mean_value() * cylinder_diameter); 

  // Reynolds number.
  const double Reynolds_number = inlet_velocity.mean_value() * cylinder_diameter / nu;

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

  // (M / deltat - theta * A)
  TrilinosWrappers::BlockSparseMatrix lhs_matrix;

  // (M / deltat - (1 - theta) * A)
  TrilinosWrappers::BlockSparseMatrix rhs_matrix;

  // A/deltat + A B^T; -B 0
  TrilinosWrappers::BlockSparseMatrix constant_matrix;

  // Pressure mass matrix, needed for preconditioning. We use a block matrix for
  // convenience, but in practice we only look at the pressure-pressure block.
  TrilinosWrappers::BlockSparseMatrix pressure_mass;

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
