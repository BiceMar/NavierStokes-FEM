#include <getopt.h>
#include "NavierStokes.hpp"
#include "NavierStokes.cpp"
void print_usage() {
    std::cout << "Usage: ./navier_stokes_solver\n --mesh <mesh_file>\n --degree_velocity <degree>\n --degree_pressure <degree>\n --T <total_time>\n --deltat <time_step>\n --theta <theta>\n --nu <viscosity>\n --p_out <pressure>\n --rho <density>\n --case_type <case_type>\n --vel <velocity>\n --prec <preconditioner>\n --dim <dim>\n";
}

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::cout << "number of processes: " << world_size << "." << std::endl;


    std::string  mesh_file_name  = "../mesh/mesh-0.05.msh";
    unsigned int degree_velocity = 2; // Default degree for velocity
    unsigned int degree_pressure = 1; // Default degree for pressure
    double T = 1.0; // Default total time: 1.0
    double deltat = 0.125; // Default time step
    double theta = 1.0; // Default theta parameter
    double nu = 0.001; // Default kinematic viscosity
    double p_out = 10.0; // Default outlet pressure
    double rho = 1.0; // Default density
    int case_type = 1; // Default case type
    double vel = 0.45; // Default velocity: 2.25
    unsigned int  prec = 0; // Default prec: 0
    unsigned int dim = 3; // Default dim: 3
        //const int d = 3;

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
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    bool show_help = false;
    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "m:V:P:T:d:t:n:p:r:c:v:e:i:h", long_options, &option_index)) != -1) {
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
                break;
            case 'd':
                deltat = std::stod(optarg);
                break;
            case 't':
                theta = std::stod(optarg);
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
                case_type = std::stoi(optarg);
                if (case_type != 0 && case_type != 1) {
                    std::cerr << "Invalid case_type: " << case_type << ". Must be either 0 or 1.\n";
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
                    std::cerr << "Invalid preconditioner: " << case_type << ". Must be either 0 or 1 or 2 or 3.\n";
                    print_usage();
                    std::exit(EXIT_FAILURE);
                }
                break;
            case 'i':
                dim = std::stoi(optarg);
                if (dim != 2 && dim != 3) {
                    std::cerr << "Invalid dimension: " << case_type << ". Must be either 2 or 3.\n";
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

    try {
        NavierStokes<3> navier_stokes(mesh_file_name, degree_velocity, degree_pressure, T, deltat, theta, nu, p_out, rho, case_type, vel, prec);

        navier_stokes.setup();
        navier_stokes.solve();

        return 0;
    } catch (std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type!\n";
    }

    return 1;
}