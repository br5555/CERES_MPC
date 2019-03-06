#include <morus_control/CERES_MPC_Autodiff.h>

CERES_MPC_AutoDiff::CERES_MPC_AutoDiff(Matrix<double, num_state_variables, num_state_variables> A, Matrix<double, num_state_variables, num_manipulated_variables>  B,
	Matrix<double, num_state_variables, num_state_variables> Bd, Matrix<double, num_state_variables, num_state_variables> Q,
	Matrix<double, num_state_variables, num_state_variables> Q_final, Matrix<double, num_manipulated_variables, num_manipulated_variables> R,
	Matrix<double, num_manipulated_variables, num_manipulated_variables> R_delta,
	Matrix<double, num_state_variables, 1> disturbance, int num_params, int pred_horizon, int control_horizon,
	Matrix<double, num_manipulated_variables, num_manipulated_variables> scale_MV, Matrix<double, num_state_variables, num_state_variables> scale_OV) : num_params_(num_params), pred_horizon(pred_horizon), control_horizon(control_horizon), A_(A), B_(B), Bd_(Bd), Q_(Q), Q_final_(Q_final), R_(R), R_delta_(R_delta), disturbance_(disturbance)
{
	//this->insecure_ = this->Bd_ * disturbance;
	
	u_current.setZero();
	u_ss_.setZero();
	
	
	x0_.setZero();
	x_ss_.setZero();
	
	
	
	count_jacobians = 0;
	scale_MV_inv = scale_MV.inverse();
	scale_OV_inv = scale_OV.inverse();
	A_pow_B_cache.setZero();
	A_pow_B_cache.block(0, 0, A_.rows(), B_.cols()) = MatrixXd::Identity(A_.rows(), A_.cols())* B_;

	for (int i = 0; i < pred_horizon - 1; i++) {

		A_pow_B_cache.block(0, (i + 1)*B_.cols(), A_.rows(), B_.cols()) = (A_* A_pow_B_cache.block(0, (i)*B_.cols(), A_.rows(), B_.cols())).eval();

	}

	du_limit(0, 0) = 70 * 1e-3 * 2;
	du_limit(0, 1) = 2;
	du_limit(1, 0) = -70 * 1e-3 * 2;
	du_limit(1, 1) = -2;
	u_limit(0, 0) = 0.6 / 2 - 0.01;
	u_limit(0, 1) = 50;
	u_limit(1, 0) = -u_limit(0, 0);
	u_limit(1, 1) = -u_limit(0, 1);
	
	m_CostFunction = new ceres::AutoDiffCostFunction<
			CERES_MPC_AutoDiff, 1 /* residuals */,
			10 /* input variables */>(this);
}



ceres::CostFunction* CERES_MPC_AutoDiff::CreateAutoDiffCostFunction() {

           

			return m_CostFunction;
			
        
        
		
	}
	
	
	
template<typename T>
	bool CERES_MPC_AutoDiff::operator()(const T* const x,
		T* const residuals) const {
		
		Matrix<T, num_state_variables, prediction_horizon>  lambdas_x;
	Matrix<T, num_manipulated_variables, prediction_horizon>  lambdas_u, lambdas_u_ref;
	Matrix<T, num_state_variables, prediction_horizon + 1>  x_states; // + 1 x0 is added 
	Matrix<T, num_manipulated_variables, mpc_control_horizon> deriv_wrt_u;
	Matrix<T, num_manipulated_variables, 1>   u, u_past;
	T residuum, residuum_signal, residuum_state;
    
	x_states.setZero();
	deriv_wrt_u.setZero();
	u.setZero();
	u_past.setZero();

	lambdas_x.setZero();
	lambdas_u.setZero();
	lambdas_u_ref.setZero();
    
    
	deriv_wrt_u.setZero();
	
	x_states.block(0, 0, x0_.rows(), x0_.cols()) = x0_.cast<T>();
    
	//u_past.block(0,0,u_current.rows(), u_current.cols())= u_current;
	u_past = T(0)*u_current.cast<T>(); //  MatrixXd::Zero(B_.cols(), 1);
    
   //bilo je horizon -1
	for (int i = 0; i < pred_horizon; i++) {
		if (i < control_horizon) {
			u << x[0 * mpc_control_horizon + (i)], x[0 * mpc_control_horizon + (i)],
				x[1 * mpc_control_horizon + (i)], -x[1 * mpc_control_horizon + (i)];

		}
		
		x_states.block(0, i + 1, x0_.rows(), x0_.cols()) = (A_.cast<T>() * x_states.block(0, i, x0_.rows(), x0_.cols()) + B_.cast<T>() * u).eval();
		lambdas_x.block(0, i, x0_.rows(), x0_.cols()) = T(-1) * x_ss_.cast<T>() + x_states.block(0, i, x0_.rows(), x0_.cols());



		lambdas_u.block(0, i, u_past.rows(), u_past.cols()) = u - u_past;
		lambdas_u_ref.block(0, i, u.rows(), u.cols()) = u - u_ss_.cast<T>();


		

		u_past = u;
	}
    
	lambdas_u_ref = scale_MV_inv.cast<T>() * lambdas_u_ref;
	lambdas_u = scale_MV_inv.cast<T>() * lambdas_u;
	lambdas_x = scale_OV_inv.cast<T>() * lambdas_x;

    //x_states_1  =  x_states.cast<double>();


	residuum_signal = (lambdas_u_ref.cwiseProduct(R_.cast<T>()*lambdas_u_ref)).sum() + (lambdas_u.cwiseProduct(R_delta_.cast<T>()*lambdas_u)).sum();

	residuum_state = (lambdas_x.cwiseProduct(Q_.cast<T>()*lambdas_x)).sum();

	residuum = residuum_signal + residuum_state;
	residuals[0] = residuum;
	return true;
	}
	
	

