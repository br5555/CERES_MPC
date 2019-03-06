
#include <ros/ros.h>
#include<std_msgs/Float64.h> 
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/src/MatrixFunctions/MatrixExponential.h>
#include <math.h>
#include <morus_control/steady_state_calculation.h> 
#include <morus_control/CERES_MPC_Autodiff.h>
#include <custom_msgs/VecOfDoubles.h> 
#include <stdio.h>
#include <thread>

constexpr int combined_control_mpc_use_ = 1;  // still working with moving masses

    // MM_MPC + added variables for CC_MPC
constexpr int kStateSize = 6 + 2*combined_control_mpc_use_; // [x1, dx1, x3, dx3, theta, dtheta] -> A is [6,6]
constexpr int kInputSize = 2 + 2*combined_control_mpc_use_; // [x1_ref (m), x3_ref (m)]          -> B is [6,2]
constexpr int kMeasurementSize = 1;                        // [theta] -> C is [1,6]
constexpr int kDisturbanceSize = kStateSize;               // disturbances are looked on all states -> B_d is [6,6]

constexpr int kControlHorizonSteps = 5;
constexpr int kPredictionHorizonSteps = 14;
constexpr double kGravity = 9.80665;



void setSolverParameterSettings1(ceres::Solver::Options* options){
        (*options).max_num_iterations = 200;
        (*options).linear_solver_type = ceres::DENSE_QR;
        //(*options).line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
        (*options).minimizer_progress_to_stdout = false;
        (*options).num_threads = std::thread::hardware_concurrency();
        (*options).initial_trust_region_radius = 1e-3;//smanjuj ako su losi -10
        (*options).max_solver_time_in_seconds = 100.050;
        (*options).gradient_tolerance=1e-8;//-4 ili -5
        (*options).parameter_tolerance=1e-8;
        (*options).function_tolerance=1e-8;
        
    
}

void setSolverParameterSettings2(ceres::Solver::Options* options){
        (*options).max_num_iterations = 200;
        (*options).linear_solver_type = ceres::DENSE_QR;
        (*options).minimizer_progress_to_stdout = false;
        (*options).num_threads = std::thread::hardware_concurrency();
        (*options).initial_trust_region_radius = 1e-4;
        (*options).max_solver_time_in_seconds = 0.050;

        /*
        (*options).use_nonmonotonic_steps=true; 
        (*options).max_lbfgs_rank=20;
        (*options).use_approximate_eigenvalue_bfgs_scaling = false;
        (*options).use_nonmonotonic_steps  =false;
        (*options).max_consecutive_nonmonotonic_steps = 5;
        (*options).initial_trust_region_radius=1e4;
        (*options).min_relative_decrease=1e-3;
        (*options).min_lm_diagonal =1e6;
        (*options).max_lm_diagonal = 1e32;
        (*options).max_num_consecutive_invalid_steps=5;
        (*options).function_tolerance=1e-6;
        (*options).gradient_tolerance=1e-10;
        (*options).parameter_tolerance=1e-8;
        (*options).update_state_every_iteration=false;*/
        
}


void setSolverParameterSettings3(ceres::Solver::Options* options){
        (*options).max_num_iterations = 200;
        (*options).linear_solver_type = ceres::DENSE_QR;
        (*options).minimizer_progress_to_stdout = false;
        (*options).num_threads = 4;
        (*options).initial_trust_region_radius = 1e4;
        (*options).max_solver_time_in_seconds = 50.0;
    
}

void setSolverParameterSettings4(ceres::Solver::Options* options){
                (*options).max_num_iterations = 200;
        (*options).linear_solver_type = ceres::DENSE_QR;
        (*options).minimizer_progress_to_stdout = false;
        (*options).num_threads = std::thread::hardware_concurrency();
        (*options).initial_trust_region_radius = 1e-5;
        (*options).max_solver_time_in_seconds = 0.050;

    
}

void setSolverParameterSettings5(ceres::Solver::Options* options){
                (*options).max_num_iterations = 200;
        (*options).linear_solver_type = ceres::DENSE_QR;
        (*options).minimizer_progress_to_stdout = false;
        (*options).num_threads = std::thread::hardware_concurrency();
        (*options).initial_trust_region_radius = 1e-6;
        (*options).max_solver_time_in_seconds = 0.050;

    
}
     

void setSolverParameterSettings6(ceres::Solver::Options* options ){
                (*options).max_num_iterations = 200;
        (*options).linear_solver_type = ceres::DENSE_QR;
        (*options).minimizer_progress_to_stdout = false;
        (*options).num_threads = std::thread::hardware_concurrency();
        (*options).initial_trust_region_radius = 1e-2;
        (*options).max_solver_time_in_seconds = 0.050;

    
}

int SolveMyOptimizationProblem(ceres::Solver::Options& options , ceres::Solver::Summary& summary , ceres::Problem& problem , Eigen::MatrixXd du_max , Eigen::MatrixXd du_min , Eigen::MatrixXd u_max , Eigen::MatrixXd u_min ,  double x[]) {

    
      Eigen::Matrix<double, 2, 1>       problem_upper_bounderies;
      Eigen::Matrix<double, 2, 1>       problem_lower_bounderies;
      cout <<"before signals are " <<x[0]<<"  "<<  x[kControlHorizonSteps]<< endl;
      problem_upper_bounderies << x[0], x[kControlHorizonSteps];
      problem_lower_bounderies << x[0], x[kControlHorizonSteps];
      
      //cout << "Prosao 1.1 " << endl;
      for(int j = 0; j< kControlHorizonSteps; j++){
            for(int i = 0; i< 2; i++){
                problem_upper_bounderies(i, 0) += du_max(i,0);
                problem_lower_bounderies(i, 0) += du_min(i,0);
                
                if(problem_upper_bounderies(i, 0) > u_max(i,0)) problem_upper_bounderies(i, 0) = u_max(i,0);
                if(problem_lower_bounderies(i, 0) < u_min(i,0)) problem_lower_bounderies(i, 0) = u_min(i,0);
                //problem_upper_bounderies(i, 0) = u_max(i,0);
                //problem_lower_bounderies(i, 0) = u_min(i,0);
                problem.SetParameterLowerBound(x,i*kControlHorizonSteps + j, problem_lower_bounderies(i, 0));
                problem.SetParameterUpperBound(x,i*kControlHorizonSteps + j, problem_upper_bounderies(i,0));
            }
      
      
      }
      //cout << "Prosao 1.2 " << endl;
      //cout << sizeof(x) << endl;
      // Run the solver!

      ceres::Solve(options, &problem, &summary);
      //cout << "Prosao 1.3 " << endl;
      vector<ceres::IterationSummary> iterations = summary.iterations;
      //cout << "time in sec " << summary.total_time_in_seconds << " num threads " << summary.num_threads_used << "num iter " << iterations.at(iterations.size()-1).iteration << endl;
      //cout <<"signals are " <<x[0]<<"  "<<  x[kControlHorizonSteps]<<"  " << endl;
      
        std::cout << summary.FullReport() << '\n';
        //ROS_INFO("Iterations : "<< ); // iterations[i].iteration);
        return iterations.at(iterations.size()-1).iteration ;

      
      std::cout << summary.BriefReport() << '\n';
      std::cout << summary.IsSolutionUsable() << '\n';
      // summary.IsSolutionUsable();
}






int main(int argc, char **argv){
//cout << "Proso -1" << endl;
ros::init(argc, argv, "minimal_publisher"); // name of this node will be "minimal_publisher"
ros::NodeHandle nh_, private_nh_; // two lines to create a publisher object that can talk to ROS
Eigen::Matrix<double, kStateSize, 1> target_state;
Eigen::Matrix<double, kStateSize, 1> current_state;
Eigen::Matrix<double, kInputSize, 1> target_input;
Eigen::Matrix<double, kStateSize, kStateSize>       model_A_, model_A_70_ms;   
Eigen::Matrix<double, kStateSize, kInputSize>       model_B_, model_B_70_ms;   
Eigen::Matrix<double, kStateSize, kDisturbanceSize> model_Bd_;
Eigen::Matrix<double, kInputSize, 1> r_command_;
Eigen::Matrix<double, kDisturbanceSize, 1> estimated_disturbances_;
Eigen::VectorXd ref(kMeasurementSize);
Eigen::Matrix<double, kStateSize, kStateSize>       A_contin;   
Eigen::Matrix<double, kStateSize, kInputSize>       B_contin;
Eigen::Matrix<double, kStateSize, kStateSize>       OV_scale;
Eigen::Matrix<double, kInputSize, kInputSize>       MV_scale;
Eigen::Matrix<double, kStateSize, kStateSize> Q;
Eigen::Matrix<double, kStateSize, kStateSize> Q_final;
Eigen::MatrixXd R;
Eigen::MatrixXd R_delta;

Eigen::Matrix<double, 2, 1>       u_max;
Eigen::Matrix<double, 2, 1>       du_max; 
Eigen::Matrix<double, 2, 1>       u_min;
Eigen::Matrix<double, 2, 1>       du_min;
double         lm_ = 0.6;
double sampling_time_(0.078684);

   // cout << "Proso 0" << endl;
current_state.setZero();
    
u_max(0,0) =  lm_/2 - 0.01; 

u_min(0,0) =  -u_max(0,0); 



du_max(0,0) =  sampling_time_ * 2; 

du_min(0,0) =  -du_max(0,0); 
 

      
       
u_max(1,0) =  50; 

u_min(1,0) =  -u_max(1,0); 


        
du_max(1,0) =  10; 
du_min(1,0) =  -10; 
cout << "Proso 1" << endl;




ref << 0;
r_command_ << 0,0 ,0, 0;

MV_scale = MatrixXd::Zero(kInputSize, kInputSize);
OV_scale = MatrixXd::Zero(kStateSize, kStateSize);
OV_scale(0,0) = 0.58;
OV_scale(1,1) = 4.0;
OV_scale(2,2) = 0.58;
OV_scale(3,3) = 4.0;
OV_scale(4,4) = 800;
OV_scale(5,5) = 800;
OV_scale(6,6) = 0.5236;
OV_scale(7,7) = 0.5236;

MV_scale(0,0) = 0.58;
MV_scale(1,1) = 0.58;
MV_scale(2,2) = 100;
MV_scale(3,3) = 100;




Q_final = 1e-5*MatrixXd::Identity(kStateSize, kStateSize);
Q(4,4) = 1e-8;
Q(5,5) = 1e-8;
Q = Q_final;
R = 1e5*MatrixXd::Identity(model_B_.cols(), model_B_.cols());//1e7
R(0,0) = 1e8;//1e10
R(1,1) = 1e8;//1e10
R_delta = R;


R(0,0) =0.67670;
R(1,1) = 0.67670;
R(2,2) = 0.13534000;
R(3,3) = 0.13534000;
   
R_delta(0,0) = 0.738879858135067 ;
R_delta(1,1) = 0.738879858135067 ;
R_delta(2,2) = 0.007388798581351;
R_delta(3,3) = 0.007388798581351;



Q(0,0) = 0.135340000000000;
Q(1,1) = 0.002706800000000;
Q(2,2) =  0.1353400;
Q(3,3) = 0.002706800;
Q(4,4) = 0.002706800;
Q(5,5) = 0.002706800;
Q(6,6) = 10.7068000;
Q(7,7) =9.676700;

A_contin   <<         0  ,        1  ,        0  ,        0 ,         0   ,       0  ,        0  ,        0,
  -391.426  ,  -25.894 ,         0   ,       0,          0   ,       0 ,         0 ,         0,
         0  ,        0 ,         0   ,       1 ,         0   ,       0 ,         0 ,         0,
         0 ,         0 ,  -391.426  ,  -25.894  ,        0   ,       0 ,         0 ,         0,
         0,          0 ,         0  ,        0  ,       -4   ,       0 ,         0 ,         0,
         0,          0 ,         0 ,         0  ,        0   ,      -4 ,         0 ,         0,
         0,          0 ,         0 ,         0  ,        0  ,        0 ,         0 ,         1,
    4.7025,   0.197406  ,   4.7025 ,  0.197406 , 0.0417255 ,-0.0417255 ,         0,          0;
    
B_contin   <<        0  ,      0 ,       0     ,   0,
 391.426,        0      ,  0,        0,
       0 ,       0     ,   0 ,       0,
       0,  391.426    ,    0  ,      0,
       0 ,       0   ,     4  ,      0,
       0 ,       0   ,     0  ,      4,
       0 ,       0  ,      0  ,      0,
-2.98409, -2.98409 ,       0  ,      0;





 model_A_ <<   0.782714 ,   0.0224343 ,           0 ,           0   ,         0,            0  ,          0,            0,
                -8.78138 ,    0.201802,            0 ,           0,            0   ,         0 ,           0      ,      0,
                       0 ,           0,     0.782714,    0.0224343,            0  ,          0  ,          0     ,       0,
                       0 ,           0,     -8.78138,     0.201802,            0 ,           0   ,         0    ,        0,
                       0 ,           0,            0,            0,     0.852144 ,           0    ,        0   ,         0,
                       0,            0,            0,            0,            0,     0.852144    ,        0  ,          0,
              0.00297324,  0.000147926,   0.00297324,  0.000147926,  3.16691e-05, -3.16691e-05     ,       1 ,        0.04,
                0.130198,   0.00703908,     0.130198,   0.00703908,   0.00154234,  -0.00154234     ,       0,            1;

model_A_   <<           1     ,   0.04,           0      ,     0 ,          0     ,      0,           0,           0,
   -15.6571  ,-0.0357581      ,     0   ,        0       ,    0    ,       0      ,     0  ,         0,
          0   ,        0     ,      1    ,    0.04      ,     0     ,      0     ,      0   ,        0,
          0    ,       0    ,-15.6571  ,-0.0357581     ,      0      ,     0    ,       0   ,        0,
          0     ,      0   ,        0   ,        0    ,    0.84       ,    0   ,        0  ,         0,
          0      ,     0  ,         0    ,       0   ,        0        ,0.84  ,         0    ,       0,
          0       ,    0 ,          0     ,      0  ,         0         ,  0 ,          1     ,   0.04,
     0.1881 , 0.00789623,      0.1881 , 0.00789623 , 0.00166902 ,-0.00166902,           0      ,     1;

model_B_ << 0.215525,            0,            0       ,     0,
             8.84393,            0 ,           0,            0,
                   0 ,    0.215525,            0,            0,
                   0 ,     8.84393 ,           0,            0,
                   0 ,           0 ,    0.147975,            0,
                   0 ,           0 ,           0,     0.147975,
         -0.00158618 , -0.00158618 , 1.68604e-06, -1.68604e-06,
          -0.0620126 ,  -0.0620126 , 0.000125442, -0.000125442;
          
model_Bd_ <<  0.0368517,  0.000550615 ,           0,            0 ,           0   ,         0,            0 ,           0,
               -0.215525,    0.0225941,            0,            0,            0 ,           0 ,           0   ,         0,
                       0,            0,    0.0368517 , 0.000550615,            0,            0 ,           0  ,          0,
                       0,            0,    -0.215525 ,   0.0225941,            0,            0 ,           0  ,          0,
                       0,            0,            0 ,           0,    0.0369936,            0 ,           0  ,          0,
                       0,            0,            0 ,           0,            0,    0.0369936 ,           0  ,          0,
             4.15876e-05,  1.98561e-06,  4.15876e-05,  1.98561e-06,   4.2151e-07,  -4.2151e-07 ,        0.04 ,    0.000792,
              0.00294716,  0.000146518,   0.00294716,  0.000146518,  3.13605e-05, -3.13605e-05 ,           0,         0.04;

estimated_disturbances_ << 0.0744648,
                            -0.00684855,
                              -0.468717,
                               0.553529,
                                     20,
                                     20,
                              -0.139467,
                            -0.00190078;
model_A_70_ms <<    0.4271,    0.0223,         0,         0,         0,         0,         0 ,        0,
   -8.7243,   -0.1501,         0,         0,         0,         0,         0,         0,
         0,         0,    0.4271,    0.0223,         0,         0,         0,         0,
         0,         0,   -8.7243,   -0.1501,         0,         0,         0,         0,
         0,         0,         0,         0,    0.7300,         0,         0,         0,
         0,         0,         0,         0,         0,    0.7300,         0,         0,
    0.0090,    0.0005,    0.0090,    0.0005,    0.0001,   -0.0001,    1.0000,    0.0787,
    0.1699,    0.0113,    0.1699,    0.0113,    0.0028,   -0.0028,         0 ,   1.0000;
    
model_B_70_ms << 0.5729,         0 ,        0 ,        0,
    8.7243,         0   ,      0 ,        0,
         0,    0.5729  ,       0 ,        0,
         0,    8.7243 ,        0 ,        0,
         0,         0 ,   0.2700 ,        0,
         0,         0 ,        0 ,   0.2700,
   -0.0037,   -0.0037,    0.0000 ,  -0.0000,
   -0.0347,   -0.0347,    0.0005 ,  -0.0005;

mav_control_attitude::SteadyStateCalculation steady_state_calculation_disk(nh_, private_nh_);
steady_state_calculation_disk.setRCommand(r_command_);
steady_state_calculation_disk.initialize(model_A_, model_B_, model_Bd_);

mav_control_attitude::SteadyStateCalculation steady_state_calculation_cont(nh_, private_nh_);
steady_state_calculation_cont.setRCommand(r_command_);
steady_state_calculation_cont.initialize(A_contin, B_contin , model_Bd_);///model_A_, model_B_, model_Bd_);

CERES_MPC_AutoDiff*  cost_disk=new CERES_MPC_AutoDiff(  model_A_70_ms,  model_B_70_ms,  model_Bd_,  Q, Q_final,
                              R,  R_delta, estimated_disturbances_, kStateSize, 14,kControlHorizonSteps , MV_scale,OV_scale);
                              
CERES_MPC_AutoDiff*  cost_cont=new CERES_MPC_AutoDiff(  model_A_70_ms,  model_B_70_ms,  model_Bd_,  Q, Q_final,
                              R,  R_delta, estimated_disturbances_, kStateSize, 14,kControlHorizonSteps , MV_scale,OV_scale);
Eigen::VectorXd v = Eigen::VectorXd::Random((2)*kControlHorizonSteps);



double x1_disk[(2)*kControlHorizonSteps] = {0};
double x2_disk[(2)*kControlHorizonSteps] = {0};
double x3_disk[(2)*kControlHorizonSteps] = {0};
double x4_disk[(2)*kControlHorizonSteps] = {0};
double x5_disk[(2)*kControlHorizonSteps] = {0};
double x6_disk[(2)*kControlHorizonSteps] = {0};


double x1_cont[(2)*kControlHorizonSteps] = {0};
double x2_cont[(2)*kControlHorizonSteps] = {0};
double x3_cont[(2)*kControlHorizonSteps] = {0};
double x4_cont[(2)*kControlHorizonSteps] = {0};
double x5_cont[(2)*kControlHorizonSteps] = {0};
double x6_cont[(2)*kControlHorizonSteps] = {0};
cout << sizeof(x1_cont)/sizeof(*x1_cont) << " velicina nakon inicijalizacije" << endl;
for(int i = 0;i < (2)*kControlHorizonSteps ; i++){
    x1_cont[i] = 0*v(i);
    cout << i << endl;
}
ceres::Problem problem1_disk,problem2_disk,problem3_disk,problem4_disk,problem5_disk,problem6_disk ;
ceres::Problem problem1_cont,problem2_cont,problem3_cont,problem4_cont,problem5_cont,problem6_cont ;
ceres::Solver::Options options;      
ceres::Solver::Summary summary;




    ros::Publisher my_publisher_m_i = nh_.advertise<std_msgs::Float64>("my_debug/m_i", 1);
    ros::Publisher my_publisher_m_i_2 = nh_.advertise<std_msgs::Float64>("my_debug/m_i_2", 1);
    ros::Publisher my_publisher_delta_omega_i = nh_.advertise<std_msgs::Float64>("my_debug/delta_omega_i", 1);
    ros::Publisher my_publisher_delta_omega_i_2 = nh_.advertise<std_msgs::Float64>("my_debug/delta_omega_i_2", 1);
    ros::Publisher my_publisher_referenaca = nh_.advertise<std_msgs::Float64>("my_debug/referenca", 1);
    
    ros::Publisher my_publisher_iteration1_disk = nh_.advertise<std_msgs::Float64>("my_debug/iteration1_disk", 1);
    ros::Publisher my_publisher_iteration2_disk = nh_.advertise<std_msgs::Float64>("my_debug/iteration2_disk", 1);
    ros::Publisher my_publisher_iteration3_disk = nh_.advertise<std_msgs::Float64>("my_debug/iteration3_disk", 1);
    ros::Publisher my_publisher_iteration4_disk = nh_.advertise<std_msgs::Float64>("my_debug/iteration4_disk", 1);
    ros::Publisher my_publisher_iteration5_disk = nh_.advertise<std_msgs::Float64>("my_debug/iteration5_disk", 1);
    ros::Publisher my_publisher_iteration6_disk = nh_.advertise<std_msgs::Float64>("my_debug/iteration6_disk", 1);
    
    ros::Publisher my_publisher_iteration1_const = nh_.advertise<std_msgs::Float64>("my_debug/iteration1_const", 1);
    ros::Publisher my_publisher_iteration2_const = nh_.advertise<std_msgs::Float64>("my_debug/iteration2_const", 1);
    ros::Publisher my_publisher_iteration3_const = nh_.advertise<std_msgs::Float64>("my_debug/iteration3_const", 1);
    ros::Publisher my_publisher_iteration4_const = nh_.advertise<std_msgs::Float64>("my_debug/iteration4_const", 1);
    ros::Publisher my_publisher_iteration5_const = nh_.advertise<std_msgs::Float64>("my_debug/iteration5_const", 1);
    ros::Publisher my_publisher_iteration6_const = nh_.advertise<std_msgs::Float64>("my_debug/iteration6_const", 1);
    
    ros::Publisher my_publisher_residuum_state_1_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_1_const", 1);
    ros::Publisher my_publisher_residuum_state_2_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_2_const", 1);
    ros::Publisher my_publisher_residuum_state_3_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_3_const", 1);
    ros::Publisher my_publisher_residuum_state_4_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_4_const", 1);
    ros::Publisher my_publisher_residuum_state_5_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_5_const", 1);
    ros::Publisher my_publisher_residuum_state_6_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_6_const", 1);
    
    ros::Publisher my_publisher_residuum_state_1_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_1_disk", 1);
    ros::Publisher my_publisher_residuum_state_2_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_2_disk", 1);
    ros::Publisher my_publisher_residuum_state_3_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_3_disk", 1);
    ros::Publisher my_publisher_residuum_state_4_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_4_disk", 1);
    ros::Publisher my_publisher_residuum_state_5_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_5_disk", 1);
    ros::Publisher my_publisher_residuum_state_6_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_state_6_disk", 1);
    
    ros::Publisher my_publisher_residuum_input_1_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_1_const", 1);
    ros::Publisher my_publisher_residuum_input_2_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_2_const", 1);
    ros::Publisher my_publisher_residuum_input_3_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_3_const", 1);
    ros::Publisher my_publisher_residuum_input_4_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_4_const", 1);
    ros::Publisher my_publisher_residuum_input_5_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_5_const", 1);
    ros::Publisher my_publisher_residuum_input_6_const = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_6_const", 1);
    
    ros::Publisher my_publisher_residuum_input_1_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_1_disk", 1);
    ros::Publisher my_publisher_residuum_input_2_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_2_disk", 1);
    ros::Publisher my_publisher_residuum_input_3_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_3_disk", 1);
    ros::Publisher my_publisher_residuum_input_4_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_4_disk", 1);
    ros::Publisher my_publisher_residuum_input_5_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_5_disk", 1);
    ros::Publisher my_publisher_residuum_input_6_disk = nh_.advertise<std_msgs::Float64>("my_debug/residuum_input_6_disk", 1);
    
    ros::Publisher my_publisher_x_i = nh_.advertise<std_msgs::Float64>("my_debug/x_i", 1);
    ros::Publisher my_publisher_dx_i = nh_.advertise<std_msgs::Float64>("my_debug/dx_i", 1);
    ros::Publisher my_publisher_x_i_2 = nh_.advertise<std_msgs::Float64>("my_debug/x_i_2", 1);
    ros::Publisher my_publisher_dx_i_2 = nh_.advertise<std_msgs::Float64>("my_debug/dx_i_2", 1);
    ros::Publisher my_publisher_omega_i = nh_.advertise<std_msgs::Float64>("my_debug/omega_i", 1);
    ros::Publisher my_publisher_omega_i_2 = nh_.advertise<std_msgs::Float64>("my_debug/omega_i_2", 1);
    ros::Publisher my_publisher_theta = nh_.advertise<std_msgs::Float64>("my_debug/theta", 1);
    ros::Publisher my_publisher_dtheta = nh_.advertise<std_msgs::Float64>("my_debug/dtheta", 1);
    
    ros::Publisher my_publisher_x1_disk = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x1_disk", 1);
    ros::Publisher my_publisher_x2_disk = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x3_disk", 1);
    ros::Publisher my_publisher_x3_disk = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x3_disk", 1);
    ros::Publisher my_publisher_x4_disk = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x4_disk", 1);
    ros::Publisher my_publisher_x5_disk = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x5_disk", 1);
    ros::Publisher my_publisher_x6_disk = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x6_disk", 1);
    
    ros::Publisher my_publisher_x1_cons = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x1_cons", 1);
    ros::Publisher my_publisher_x2_cons = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x2_cons", 1);
    ros::Publisher my_publisher_x3_cons = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x3_cons", 1);
    ros::Publisher my_publisher_x4_cons = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x4_cons", 1);
    ros::Publisher my_publisher_x5_cons = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x5_cons", 1);
    ros::Publisher my_publisher_x6_cons = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/x6_cons", 1);
    
    ros::Publisher my_publisher_traget_state_const = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/traget_state_const", 1);
    ros::Publisher my_publisher_traget_state_disk = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/traget_state_disk", 1);
    ros::Publisher my_publisher_traget_input_const = nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/traget_input_const", 1);
    ros::Publisher my_publisher_traget_input_disk= nh_.advertise<custom_msgs::VecOfDoubles>("my_debug/traget_input_disk", 1);
    
 
    
    custom_msgs::VecOfDoubles traget_input_const;
    traget_input_const.dbl_vec.resize(kInputSize);
    
    custom_msgs::VecOfDoubles traget_input_disk;
    traget_input_disk.dbl_vec.resize(kInputSize);
    
    custom_msgs::VecOfDoubles traget_state_const;
    traget_state_const.dbl_vec.resize(kStateSize);
    
    custom_msgs::VecOfDoubles traget_state_disk;
    traget_state_disk.dbl_vec.resize(kStateSize);
    
    custom_msgs::VecOfDoubles vec_x1_disk;
    vec_x1_disk.dbl_vec.resize(kInputSize*kControlHorizonSteps);
    
    custom_msgs::VecOfDoubles vec_x2_disk;
    vec_x2_disk.dbl_vec.resize(kInputSize*kControlHorizonSteps);
    
    custom_msgs::VecOfDoubles vec_x3_disk;
    vec_x3_disk.dbl_vec.resize(kInputSize*kControlHorizonSteps);
    
    custom_msgs::VecOfDoubles vec_x4_disk;
    vec_x4_disk.dbl_vec.resize(kInputSize*kControlHorizonSteps);
    
    custom_msgs::VecOfDoubles vec_x5_disk;
    vec_x5_disk.dbl_vec.resize(kInputSize*kControlHorizonSteps);
    
    custom_msgs::VecOfDoubles vec_x6_disk;
    vec_x6_disk.dbl_vec.resize(kInputSize*kControlHorizonSteps);
    
    custom_msgs::VecOfDoubles vec_x1_cons;
    vec_x1_cons.dbl_vec.resize(kInputSize*kControlHorizonSteps);
    
    custom_msgs::VecOfDoubles vec_x2_cons;
    vec_x2_cons.dbl_vec.resize(kInputSize*kControlHorizonSteps);
    
    custom_msgs::VecOfDoubles vec_x3_cons;
    vec_x3_cons.dbl_vec.resize(kInputSize*kControlHorizonSteps);
    
    custom_msgs::VecOfDoubles vec_x4_cons;
    vec_x4_cons.dbl_vec.resize(kInputSize*kControlHorizonSteps);
    
    custom_msgs::VecOfDoubles vec_x5_cons;
    vec_x5_cons.dbl_vec.resize(kInputSize*kControlHorizonSteps);
    
    custom_msgs::VecOfDoubles vec_x6_cons;
    vec_x6_cons.dbl_vec.resize(kInputSize*kControlHorizonSteps);
        
    std_msgs::Float64 m_i, m_i_2, delta_omega_i ,delta_omega_i_2, referenca;
    std_msgs::Float64 x_i, x_i_2, dx_i, dx_i_2, omega_i,omega_i_2, theta, dtheta; 
    std_msgs::Float64 iteration1_disk, iteration2_disk,iteration3_disk ,iteration4_disk ,iteration5_disk ,iteration6_disk;
    std_msgs::Float64 iteration1_cont, iteration2_cont,iteration3_cont ,iteration4_cont ,iteration5_cont ,iteration6_cont;
    std_msgs::Float64 residuum_input_1_const, residuum_input_2_const, residuum_input_3_const, residuum_input_4_const, residuum_input_5_const, residuum_input_6_const;
    std_msgs::Float64 residuum_input_1_disk, residuum_input_2_disk, residuum_input_3_disk, residuum_input_4_disk, residuum_input_5_disk, residuum_input_6_disk;
    std_msgs::Float64 residuum_state_1_disk, residuum_state_2_disk, residuum_state_3_disk, residuum_state_4_disk, residuum_state_5_disk, residuum_state_6_disk;
    std_msgs::Float64 residuum_state_1_cont, residuum_state_2_cont, residuum_state_3_cont, residuum_state_4_cont, residuum_state_5_cont, residuum_state_6_cont;
    cout << "Proso 2" << endl;
    ros::Rate naptime(1.0);
    

    double input_sin =  0.1397263;
    double sign = 1.0;

    while (ros::ok()) 
    {
        Eigen::Matrix<double, (4), 1>       last_control_signal;
        last_control_signal << x1_disk[0], x1_disk[0], x1_disk[kControlHorizonSteps], -x1_disk[kControlHorizonSteps];
      
       for(int i = 0; i< kControlHorizonSteps-1; i++){
            x1_cont[0 + i] =  x1_cont[ i+1]*0;
             x1_cont[kControlHorizonSteps + i] = x1_cont[kControlHorizonSteps +i +1]*0;
              
          }
 
        
        ref(0) = input_sin;// sin(input_sin);
        steady_state_calculation_disk.computeSteadyState(estimated_disturbances_, ref,
                                                    &target_state, &target_input);
                                                    
       for(int i = 0; i <kInputSize ; i++){
            traget_input_disk.dbl_vec[i] = target_input(i,0);
       }
       
       for(int i = 0; i <kStateSize ; i++){
            traget_state_disk.dbl_vec[i] = target_state(i,0);
       }
       
        
        
        
        cost_disk->set_disturbance(estimated_disturbances_);
        cost_disk->set_x_ss(target_state);
        cost_disk->set_u_ss(target_input);
        cost_disk->set_u_current(last_control_signal);
        cost_disk->set_x0_(current_state);
        
        
        
                                                    
        problem1_disk.AddResidualBlock(cost_disk->CreateAutoDiffCostFunction(), NULL, x1_disk);
        problem2_disk.AddResidualBlock(cost_disk->CreateAutoDiffCostFunction(), NULL, x2_disk);
        problem3_disk.AddResidualBlock(cost_disk->CreateAutoDiffCostFunction(), NULL, x3_disk);
        problem4_disk.AddResidualBlock(cost_disk->CreateAutoDiffCostFunction(), NULL, x4_disk);
        problem5_disk.AddResidualBlock(cost_disk->CreateAutoDiffCostFunction(), NULL, x5_disk);
        problem6_disk.AddResidualBlock(cost_disk->CreateAutoDiffCostFunction(), NULL, x6_disk);

        /*
        
        setSolverParameterSettings1(&options);
        iteration1_disk.data = SolveMyOptimizationProblem(options, summary , problem1_disk, du_max, du_min,  u_max,  u_min,  x1_disk);
        residuum_input_1_disk.data = cost_disk->get_residuum_signal();
        residuum_state_1_disk.data = cost_disk->get_residuum_state();
        
        setSolverParameterSettings2(&options);
        iteration2_disk.data = SolveMyOptimizationProblem(options, summary , problem2_disk, du_max, du_min,  u_max,  u_min,  x2_disk);
        residuum_input_2_disk.data = cost_disk->get_residuum_signal();
        residuum_state_2_disk.data = cost_disk->get_residuum_state();
        
        setSolverParameterSettings3(&options);
        iteration3_disk.data = SolveMyOptimizationProblem(options, summary , problem3_disk, du_max, du_min,  u_max,  u_min,  x3_disk);
        residuum_input_3_disk.data = cost_disk->get_residuum_signal();
        residuum_state_3_disk.data = cost_disk->get_residuum_state();
        
        setSolverParameterSettings4(&options);
        iteration4_disk.data = SolveMyOptimizationProblem(options, summary , problem4_disk, du_max, du_min,  u_max,  u_min,  x4_disk);
        residuum_input_4_disk.data = cost_disk->get_residuum_signal();
        residuum_state_4_disk.data = cost_disk->get_residuum_state();
        
        setSolverParameterSettings5(&options);
        iteration5_disk.data = SolveMyOptimizationProblem(options, summary , problem5_disk, du_max, du_min,  u_max,  u_min,  x5_disk);
        residuum_input_5_disk.data = cost_disk->get_residuum_signal();
        residuum_state_5_disk.data = cost_disk->get_residuum_state();
        
        setSolverParameterSettings6(&options);
        iteration6_disk.data = SolveMyOptimizationProblem(options, summary , problem6_disk, du_max, du_min,  u_max,  u_min,  x6_disk);
        residuum_input_6_disk.data = cost_disk->get_residuum_signal();
        residuum_state_6_disk.data = cost_disk->get_residuum_state();*/
        

        
        for(int i = 0; i < (2)*kControlHorizonSteps; i++ ){
            vec_x1_disk.dbl_vec[i] = x1_disk[i];
            vec_x2_disk.dbl_vec[i] = x2_disk[i];
            vec_x3_disk.dbl_vec[i] = x3_disk[i];
            vec_x4_disk.dbl_vec[i] = x4_disk[i];
            vec_x5_disk.dbl_vec[i] = x5_disk[i];
            vec_x6_disk.dbl_vec[i] = x6_disk[i];
            
        }
        
        

        
        target_state.setZero();
        target_input.setZero();
        target_state(6,0) = ref(0);
        
        cout << "target states " << target_state << endl;
        cout << endl;
        cout << "target input " << target_input << endl;
        cout << endl;
        cost_cont->set_disturbance(estimated_disturbances_);
        cost_cont->set_x_ss(target_state);
        cost_cont->set_u_ss(target_input);
        cost_cont->set_u_current(last_control_signal);
        cost_cont->set_x0_(current_state);
        cout << sizeof(x1_cont)/sizeof(*x1_cont) << "prije problem bloka " << endl;
        problem1_cont.AddResidualBlock(cost_cont->CreateAutoDiffCostFunction(), NULL, x1_cont);
        problem2_cont.AddResidualBlock(cost_cont->CreateAutoDiffCostFunction(), NULL, x2_cont);
        problem3_cont.AddResidualBlock(cost_cont->CreateAutoDiffCostFunction(), NULL, x3_cont);
        problem4_cont.AddResidualBlock(cost_cont->CreateAutoDiffCostFunction(), NULL, x4_cont);
        problem5_cont.AddResidualBlock(cost_cont->CreateAutoDiffCostFunction(), NULL, x5_cont);
        problem6_cont.AddResidualBlock(cost_cont->CreateAutoDiffCostFunction(), NULL, x6_cont);
        
        setSolverParameterSettings1(&options);
        
        for(int k = 0; k< 100; k++){
                
            iteration1_cont.data = SolveMyOptimizationProblem(options, summary , problem1_cont, du_max, du_min,  u_max,  u_min,  x1_cont);
            //residuum_input_1_const.data = cost_cont->get_residuum_signal();
            //residuum_state_1_cont.data = cost_cont->get_residuum_state();
            //cout << "residuum_state/ residuum_input = " <<   residuum_state_1_cont.data/residuum_input_1_const.data <<" residuum= " << residuum_state_1_cont.data +residuum_input_1_const.data<<endl; 
            cout << endl;
            cout << "k je " << k << endl;
            cout << endl;
            cout << "x_states" << endl;
            cout << (cost_cont->get_x_states()).rows() << " " << (cost_cont->get_x_states()).cols() <<endl;
            cout << (cost_cont->get_x_states()).block(0,1,8, 1) <<endl;
           cout << endl;
           cout << "target states " << target_state << endl;
        cout << endl;
            cout << endl;
            last_control_signal << x1_cont[kControlHorizonSteps*0],x1_cont[kControlHorizonSteps*0], x1_cont[kControlHorizonSteps*1],-x1_cont[kControlHorizonSteps*1];
            for(int i = 0; i< kControlHorizonSteps-1; i++){
                x1_cont[0 + i] =  x1_cont[ i+1];
                 x1_cont[kControlHorizonSteps + i] = x1_cont[kControlHorizonSteps +i +1];
                  
              }
            
            cost_cont->set_u_current(last_control_signal);
            current_state = (cost_cont->get_x_states()).block(0,1,8, 1);
            cost_cont->set_x0_(current_state);
            naptime.sleep();
        
        }
      /* setSolverParameterSettings2(&options);
        iteration2_cont.data = SolveMyOptimizationProblem(options, summary , problem2_cont, du_max, du_min,  u_max,  u_min,  x2_cont);
        residuum_input_2_const.data = cost_cont->get_residuum_signal();
        residuum_state_2_cont.data = cost_cont->get_residuum_state();
        cout << "enough time residuum_state/ residuum_input = " <<   residuum_state_2_cont.data/residuum_input_2_const.data <<" residuum= " << residuum_state_2_cont.data +residuum_input_2_const.data<<endl;
        
        
        setSolverParameterSettings3(&options);
        iteration3_cont.data = SolveMyOptimizationProblem(options, summary , problem3_cont, du_max, du_min,  u_max,  u_min,  x3_cont);
        residuum_input_3_const.data = cost_cont->get_residuum_signal();
        residuum_state_3_cont.data = cost_cont->get_residuum_state();
        cout << "enough time residuum_state/ residuum_input = " <<   residuum_state_3_cont.data/residuum_input_3_const.data <<" residuum= " << residuum_state_3_cont.data +residuum_input_3_const.data<<endl;
        
        setSolverParameterSettings4(&options);
        iteration4_cont.data = SolveMyOptimizationProblem(options, summary , problem4_cont, du_max, du_min,  u_max,  u_min,  x4_cont);
        residuum_input_4_const.data = cost_cont->get_residuum_signal();
        residuum_state_4_cont.data = cost_cont->get_residuum_state();
        cout << "enough time residuum_state/ residuum_input = " <<   residuum_state_4_cont.data/residuum_input_4_const.data <<" residuum= " << residuum_state_4_cont.data +residuum_input_4_const.data<<endl;
        
        setSolverParameterSettings5(&options);
        iteration5_cont.data = SolveMyOptimizationProblem(options, summary , problem5_cont, du_max, du_min,  u_max,  u_min,  x5_cont);
        residuum_input_5_const.data = cost_cont->get_residuum_signal();
        residuum_state_5_cont.data = cost_cont->get_residuum_state();
        cout << "enough time residuum_state/ residuum_input = " <<   residuum_state_5_cont.data/residuum_input_5_const.data <<" residuum= " << residuum_state_5_cont.data +residuum_input_5_const.data<<endl;
        
        
        setSolverParameterSettings6(&options);
        iteration6_cont.data = SolveMyOptimizationProblem(options, summary , problem6_cont, du_max, du_min,  u_max,  u_min,  x6_cont); 
        residuum_input_6_const.data = cost_cont->get_residuum_signal();
        residuum_state_6_cont.data = cost_cont->get_residuum_state();
        cout << "enough time residuum_state/ residuum_input = " <<   residuum_state_6_cont.data/residuum_input_6_const.data <<" residuum= " << residuum_state_6_cont.data +residuum_input_6_const.data<<endl;*/
        
        for(int i = 0; i < (2)*kControlHorizonSteps; i++ ){
            vec_x1_cons.dbl_vec[i] = x1_cont[i];
            vec_x2_cons.dbl_vec[i] = x2_cont[i];
            vec_x3_cons.dbl_vec[i] = x3_cont[i];
            vec_x4_cons.dbl_vec[i] = x4_cont[i];
            vec_x5_cons.dbl_vec[i] = x5_cont[i];
            vec_x6_cons.dbl_vec[i] = x6_cont[i];   
            
        }
 
 
        if(input_sin > 0.1396263){ sign *= -1.0; }
        else if(input_sin < -0.1396263) {sign *= -1.0;}
        input_sin += sign*0.01745329*17;
        
        m_i.data = target_input(0,0); 
         m_i_2.data = target_input(1,0);
          delta_omega_i.data = target_input(2,0);
          delta_omega_i_2.data=target_input(3,0); 
        referenca.data = ref(0);
        x_i.data = target_state(0,0);
        x_i_2.data = target_state(1,0);
        dx_i.data = target_state(2,0);
        dx_i_2.data = target_state(3,0);
        omega_i.data = target_state(4,0);
        omega_i_2.data = target_state(5,0);
        theta.data = target_state(6,0);
        dtheta.data = target_state(7,0);
             
        my_publisher_m_i.publish(m_i);
        my_publisher_m_i_2.publish(m_i_2);
        my_publisher_delta_omega_i.publish(delta_omega_i);
        my_publisher_delta_omega_i_2.publish(delta_omega_i_2);
        my_publisher_x_i.publish(x_i);
        my_publisher_dx_i.publish(dx_i);
        my_publisher_x_i_2.publish(x_i_2);
        my_publisher_dx_i_2.publish(dx_i_2);
        my_publisher_omega_i.publish(omega_i);
        my_publisher_omega_i_2.publish(omega_i_2);
        my_publisher_referenaca.publish(referenca);
        my_publisher_theta.publish(theta);
        my_publisher_dtheta.publish(dtheta);
        
        my_publisher_x1_disk.publish( vec_x1_disk );
        my_publisher_x2_disk.publish( vec_x2_disk );
        my_publisher_x3_disk.publish( vec_x3_disk );
        my_publisher_x4_disk.publish( vec_x4_disk );
        my_publisher_x5_disk.publish( vec_x5_disk );
        my_publisher_x6_disk.publish( vec_x6_disk );
        
        my_publisher_x1_cons.publish( vec_x1_cons );
        my_publisher_x2_cons.publish( vec_x2_cons );
        my_publisher_x3_cons.publish( vec_x3_cons );
        my_publisher_x4_cons.publish( vec_x4_cons );
        my_publisher_x5_cons.publish( vec_x5_cons );
        my_publisher_x6_cons.publish( vec_x6_cons );
        
        my_publisher_iteration1_disk.publish( iteration1_disk);
        my_publisher_iteration2_disk.publish( iteration2_disk);
        my_publisher_iteration3_disk.publish( iteration3_disk);
        my_publisher_iteration4_disk.publish( iteration4_disk);
        my_publisher_iteration5_disk.publish( iteration5_disk);
        my_publisher_iteration6_disk.publish( iteration6_disk);
        
        my_publisher_iteration1_const.publish( iteration1_cont);
        my_publisher_iteration2_const.publish( iteration2_cont);
        my_publisher_iteration3_const.publish( iteration3_cont);
        my_publisher_iteration4_const.publish( iteration4_cont);
        my_publisher_iteration5_const.publish( iteration5_cont);
        my_publisher_iteration6_const.publish( iteration6_cont);
        
        my_publisher_residuum_state_1_const.publish(residuum_state_1_cont);
        my_publisher_residuum_state_2_const.publish(residuum_state_2_cont);
        my_publisher_residuum_state_3_const.publish(residuum_state_3_cont);
        my_publisher_residuum_state_4_const.publish(residuum_state_4_cont);
        my_publisher_residuum_state_5_const.publish(residuum_state_5_cont);
        my_publisher_residuum_state_6_const.publish(residuum_state_6_cont);
                
        my_publisher_residuum_state_1_disk.publish(residuum_state_1_disk);
        my_publisher_residuum_state_2_disk.publish(residuum_state_2_disk);
        my_publisher_residuum_state_3_disk.publish(residuum_state_3_disk);
        my_publisher_residuum_state_4_disk.publish(residuum_state_4_disk);
        my_publisher_residuum_state_5_disk.publish(residuum_state_5_disk);
        my_publisher_residuum_state_6_disk.publish(residuum_state_6_disk);
    
    
        my_publisher_residuum_input_1_const.publish(residuum_input_1_const);
        my_publisher_residuum_input_2_const.publish(residuum_input_2_const);
        my_publisher_residuum_input_3_const.publish(residuum_input_3_const);
        my_publisher_residuum_input_4_const.publish(residuum_input_4_const);
        my_publisher_residuum_input_5_const.publish(residuum_input_5_const);
        my_publisher_residuum_input_6_const.publish(residuum_input_6_const);
    
    
        my_publisher_residuum_input_1_disk.publish(residuum_input_1_disk);
        my_publisher_residuum_input_2_disk.publish(residuum_input_2_disk);
        my_publisher_residuum_input_3_disk.publish(residuum_input_3_disk);
        my_publisher_residuum_input_4_disk.publish(residuum_input_4_disk);
        my_publisher_residuum_input_5_disk.publish(residuum_input_5_disk);
        my_publisher_residuum_input_6_disk.publish(residuum_input_6_disk);
        
        my_publisher_traget_state_const.publish(traget_state_const); 
        my_publisher_traget_state_disk.publish(traget_state_disk);
        my_publisher_traget_input_const.publish(traget_input_const);
        my_publisher_traget_input_disk.publish(traget_input_disk);
        

                                                                                                  
        naptime.sleep();
}



                                                   
}
