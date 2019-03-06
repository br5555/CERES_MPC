#ifndef ROS_MPC_CERES
#define ROS_MPC_CERES

#include <vector>
#include "ceres/ceres.h"
//#include "gflags/gflags.h"
#include "glog/logging.h"
#include <iostream>
#include <stdio.h>
#include <ros/ros.h> 
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/src/MatrixFunctions/MatrixExponential.h>

using namespace Eigen;
using namespace std;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using ceres::CostFunction;
using ceres::SizedCostFunction;


class MPC_cost : public SizedCostFunction<1 /* number of residuals */,
                             10/* size of first parameter (2 * horizon)*/ >{

public:
    MPC_cost(){};
    MPC_cost( MatrixXd A, MatrixXd B, MatrixXd Bd, MatrixXd Q,
                       MatrixXd Q_final, MatrixXd R, MatrixXd R_delta,
                        MatrixXd disturbance,int num_params, int pred_horizon,int control_horizon , MatrixXd scale_MV,MatrixXd scale_OV);


  virtual bool Evaluate(double const* const* x,
                                   double* residuals,
                                   double** jacobians) const;
    int dim_X(void) const{return 12;}
    void set_u_past(MatrixXd u_past_) {this->u_past = u_past_;}
    void set_u_current(MatrixXd u_current_) {this->u_current = u_current_;}
    void set_u_ss(MatrixXd u_ss) {this->u_ss_ = u_ss;}
    void set_x_ss(MatrixXd x_ss) {this->x_ss_ = x_ss;}
    void set_x0_(MatrixXd x0) {this->x0_ = x0;}
    void set_A(MatrixXd A){this->A_ = A;}
    void set_B(MatrixXd B){this->B_ = B;}
    void set_Bd(MatrixXd Bd){this->Bd_ = Bd;}
    void set_Q(MatrixXd Q){this->Q_ = Q;}
    void set_Q_final(MatrixXd Q_final){this->Q_final_ = Q_final;}
    void set_R(MatrixXd R){this->R_ = R;}
    void set_R_delta(MatrixXd R_delta){this->R_delta_ = R_delta;}
    void set_insecure(MatrixXd insecure){this->insecure_ = insecure;}
    void set_disturbance(MatrixXd disturbance){this->disturbance_ = disturbance;
                                                this->insecure_ = this->Bd_*this->disturbance_;}
    void set_num_params(int num_params){this->num_params_ = num_params;}
    double get_residuum(){return residuum;}
    double get_residuum_signal(){return residuum_signal;}
    double get_residuum_state(){return residuum_state;}
    MatrixXd get_x_states(){return x_states;}
    
    
    private:
        mutable int  num_params_, pred_horizon, control_horizon;
        mutable double residuum, residuum_signal, residuum_state;
        mutable MatrixXd A_, B_,Bd_, Q_, Q_final_,  R_, R_delta_, disturbance_, insecure_, u_ss_, x_ss_, x0_, u_prev_, x_states, u,  deriv_wrt_u, u_past, lambdas_x, lambdas_u,lambdas_u_ref, u_horizon, u_current, scale_MV_inv, scale_OV_inv, A_pow_matrix, A_pow_B_cache; 

        
        
};

#endif 
