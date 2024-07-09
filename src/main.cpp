#include <getopt.h>
#include "NavierStokes.hpp"
#include "NavierStokes.cpp"

void print_usage() {
    std::cout << "Usage: ./navier_stokes_solver\n --mesh <mesh_file> : mesh-0.1, mesh-0.05, mesh-0.025, mesh-0.0125\n --degree_velocity <degree>\n --degree_pressure <degree>\n --T <total_time>\n --deltat <time_step>\n  --theta <theta>\n --nu <viscosity>\n --p_out <pressure>\n --rho <density>\n --velocity_case_type <case_type>\n --vel <velocity>\n --prec <preconditioner>: 0 for block-diagonal, 1 for SIMPLE, 2 for aSIMPLE, 3 for Yosida\n --dim <dim>: 2 for 2D, 3 for 3D\n --use_skew <use skew symmetric representation>: 0 for skew, 1 for actual non-linear term\n";
}

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::string  mesh_file_name  = "../mesh/mesh-0.025.msh";
    unsigned int degree_velocity = 2; // Default degree for velocity
    unsigned int degree_pressure = 1; // Default degree for pressure
    double T = 1.0; // Default total time: 1.0
    double deltat = 0.125; // Default time step
    double theta = 1.0; // Default theta parameter
    double nu = 0.001; // Default kinematic viscosity
    double p_out = 0.0; // Default outlet pressure
    double rho = 1.0; // Default density
    unsigned int velocity_case_type = 1; // Default case type
    double vel = 0.45; // Default velocity: 0.45
    unsigned int  prec = 0; // Default prec: 0
    unsigned int dim = 3; // Default dim: 3
    unsigned int use_skew = 0; // various representations of nonlinear term

    struct option long_options[] = {
        {"mesh", required_argument, 0, 'm'},
        {"degree_velocity", required_argument, 0, 'V'},
        {"degree_pressure", required_argument, 0, 'P'},
        {"T", required_argument, 0, 'T'},
        {"deltat", required_argument, 0, 'd'},
        {"theta", required_argument, 0, 't'},
        {"nu", required_argument, 0, 'n'},
        {"p_out", required_argument, 0, 'p'},
        {"rho", required_argument, 0, 'r'},
        {"case_type", required_argument, 0, 'c'},  
        {"vel", required_argument, 0, 'v'},  
        {"prec", required_argument, 0, 'e'},
        {"dim", required_argument, 0, 'i'},
        {"use_skew", required_argument, 0, 'u'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    bool show_help = false;
    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "m:V:P:T:d:t:n:p:r:c:v:e:i:u:h", long_options, &option_index)) != -1) {
        switch (c) {
            case 'm':
                mesh_file_name = optarg;
                break;
            case 'V':
                degree_velocity = std::stoi(optarg);
                break;
            case 'P':
                degree_pressure = std::stoi(optarg);
                break;
            case 'T':
                T = std::stod(optarg);
                if (T <= 0) {
                    std::cerr << "Invalid time: " << T << ". Must be greater than 0.\n";
                    print_usage();
                    std::exit(EXIT_FAILURE);
                }
                break;
            case 'd':
                deltat = std::stod(optarg);
                if (deltat <= 0) {
                    std::cerr << "Invalid deltat: " << deltat << ". Must be greater than 0.\n";
                    print_usage();
                    std::exit(EXIT_FAILURE);
                }
                break;
            case 't':
                theta = std::stod(optarg);
                if (theta < 0.0 || theta > 1.0) {
                    std::cerr << "Invalid theta: " << theta << ". Must be in [0.0, 1.0].\n";
                    print_usage();
                    std::exit(EXIT_FAILURE);
                }
                break;
                break;
            case 'n':
                nu = std::stod(optarg);
                break;
            case 'p':
                p_out = std::stod(optarg);
                break;
            case 'r':
                rho = std::stod(optarg);
                break;
            case 'c':
                velocity_case_type = std::stoi(optarg);
                if (velocity_case_type != 2 && velocity_case_type != 1) {
                    std::cerr << "Invalid velocity case type: " << velocity_case_type << ". Must be either 2 or 1.\n";
                    print_usage();
                    std::exit(EXIT_FAILURE);
                }
            break;
            case 'v':
                vel = std::stod(optarg);
                break;
            case 'e':
                prec = std::stoi(optarg);
                if (prec > 3) {
                    std::cerr << "Invalid preconditioner: " << prec << ". Must be either 0 or 1 or 2 or 3.\n";
                    print_usage();
                    std::exit(EXIT_FAILURE);
                }
                break;
            case 'i':
                dim = std::stoi(optarg);
                if (dim != 2 && dim != 3) {
                    std::cerr << "Invalid dimension: " << dim << ". Must be either 2 or 3.\n";
                    print_usage();
                    std::exit(EXIT_FAILURE);
                }
                break;
            case 'u':
                use_skew = std::stoi(optarg);
                if (use_skew != 0 && use_skew != 1) {
                    std::cerr << "Invalid use_skew: " << use_skew << ". Must be either 0 or 1.\n";
                    print_usage();
                    std::exit(EXIT_FAILURE);
                }
                break;
            case 'h':
                show_help = true;
                break;
            case '?':
                show_help = true;
                break;
            default:
                show_help = true;
                break;
        }
    }

    if (show_help) {
        print_usage();
        return 0;
    }

    // 2D case
    if(dim == 2){
        try {
            NavierStokes<2> navier_stokes(mesh_file_name, degree_velocity, degree_pressure, T, deltat, theta, nu, p_out, rho, velocity_case_type, vel, prec, use_skew);

            navier_stokes.setup();
            navier_stokes.solve();

            return 0;

        } catch (std::exception &e) {
            std::cerr << "error: " << e.what() << "\n";
            return 1;
        } 
    }
    
    // 3D case
    if(dim == 3){
        try {
            NavierStokes<3> navier_stokes(mesh_file_name, degree_velocity, degree_pressure, T, deltat, theta, nu, p_out, rho, velocity_case_type, vel, prec, use_skew);

            navier_stokes.setup();
            navier_stokes.solve();

            return 0;

        } catch (std::exception &e) {
            std::cerr << "error: " << e.what() << "\n";
            return 1;
        } 
    }

    return 1;
}