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


constexpr int num_state_variables = 8;
constexpr int num_manipulated_variables = 4;
constexpr int num_heuristic_variables = 2; //using simetric u1 = u2 and u3 = -u4 
constexpr int mpc_control_horizon = 5;
constexpr int prediction_horizon = 14;
constexpr int num_cost_functions = 1;

class MPC_cost : public SizedCostFunction<1 /* number of residuals */,
                             10/* size of first parameter (2 * horizon)*/ >{

public:
    MPC_cost(){};
	/*If you define a structure having members of fixed - size vectorizable Eigen types, you must overload its "operator new" so that it generates 16 - bytes - aligned pointers.*/
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	MPC_cost(Matrix<double, num_state_variables, num_state_variables> A, Matrix<double, num_state_variables, num_manipulated_variables>  B,
			Matrix<double, num_state_variables, num_state_variables> Bd, Matrix<double, num_state_variables, num_state_variables> Q,
			Matrix<double, num_state_variables, num_state_variables> Q_final, Matrix<double, num_manipulated_variables, num_manipulated_variables> R,
			Matrix<double, num_manipulated_variables, num_manipulated_variables> R_delta,
			Matrix<double, num_state_variables, 1> disturbance, int num_params, int pred_horizon, int control_horizon,
			Matrix<double, num_manipulated_variables, num_manipulated_variables> scale_MV, Matrix<double, num_state_variables, num_state_variables> scale_OV);


  virtual bool Evaluate(double const* const* x,
                                   double* residuals,
                                   double** jacobians) const;
  int dim_X(void) const { return 12; }
  //void set_u_past(Matrix<double, num_manipulated_variables, 1> u_past_) { this->u_past = u_past_; }
  void set_u_current(Matrix<double, num_manipulated_variables, 1> u_current_) { this->u_current = u_current_; }
  void set_u_ss(Matrix<double, num_manipulated_variables, 1> u_ss) { this->u_ss_ = u_ss; }
  void set_x_ss(Matrix<double, num_state_variables, 1>  x_ss) { this->x_ss_ = x_ss; new_desire_state = true; }
  void set_x0_(Matrix<double, num_state_variables, 1>  x0) { this->x0_ = x0; }
  void set_A(Matrix<double, num_state_variables, num_state_variables> A) { this->A_ = A; }
  void set_B(Matrix<double, num_state_variables, num_manipulated_variables> B) { this->B_ = B; }
  void set_Bd(Matrix<double, num_state_variables, num_state_variables> Bd) { this->Bd_ = Bd; }
  void set_Q(Matrix<double, num_state_variables, num_state_variables>  Q) { this->Q_ = Q; }
  void set_Q_final(Matrix<double, num_state_variables, num_state_variables>  Q_final) { this->Q_final_ = Q_final; }
  void set_R(Matrix<double, num_manipulated_variables, num_manipulated_variables>  R) { this->R_ = R; }
  void set_R_delta(Matrix<double, num_manipulated_variables, num_manipulated_variables>  R_delta) { this->R_delta_ = R_delta; }
  void set_insecure(Matrix<double, num_state_variables, 1> insecure) { this->insecure_ = insecure; }
  void set_disturbance(Matrix<double, num_state_variables, 1> disturbance) {
	  this->disturbance_ = disturbance;
	  this->insecure_ = this->Bd_*this->disturbance_;
  }
  void set_num_params(int num_params) { this->num_params_ = num_params; }
  //double get_residuum() { return residuum; }
  //double get_residuum_signal() { return residuum_signal; }
  //double get_residuum_state() { return residuum_state; }
  MatrixXd get_x_states() { return x_states_1; }


private:
	//Matrix<double, 2 * mpc_control_horizon, 1>&  check_bounderies(Matrix<double, 2 * mpc_control_horizon, 1>&   x);
	bool new_desire_state;
	int  num_params_, pred_horizon, control_horizon, max_iter, saturation_count, count_jacobians;
	mutable Matrix<double, num_state_variables, prediction_horizon + 1>  x_states_1;
	Matrix<double, num_state_variables, num_state_variables> A_, Bd_;
	Matrix<double, num_state_variables, num_manipulated_variables> B_;
	Matrix<double, num_manipulated_variables, num_manipulated_variables> R_, R_delta_, scale_MV_inv;
	Matrix<double, num_state_variables, num_state_variables> Q_, Q_final_, scale_OV_inv;
	Matrix<double, num_state_variables, 1> x_ss_, disturbance_, insecure_, x0_;
	Matrix<double, num_manipulated_variables, 1> u_ss_, u_prev_, u_current;
	Matrix<double, num_state_variables, prediction_horizon * num_manipulated_variables> A_pow_B_cache;
	
	Matrix<double, 2, num_heuristic_variables>  du_limit, u_limit; //2 for upper and lower

        
        
};

#endif 
