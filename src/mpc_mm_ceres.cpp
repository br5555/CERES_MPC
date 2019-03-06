#include <morus_control/mpc_mm_ceres.h>

MPC_cost::MPC_cost(Matrix<double, num_state_variables, num_state_variables> A, Matrix<double, num_state_variables, num_manipulated_variables>  B,
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
}

//When you add the const keyword to a method the this pointer will essentially become a pointer to const object, and you cannot therefore change any member data. (This is not completely true, because you can mark a member as mutable and a const method can then change it. It's mostly used for internal counters and stuff.).

bool MPC_cost::Evaluate(double const* const* x,
	double* residuals,
	double** jacobians) const {


	Matrix<double, num_state_variables, prediction_horizon>  lambdas_x;
	Matrix<double, num_manipulated_variables, prediction_horizon>  lambdas_u, lambdas_u_ref;
	Matrix<double, num_state_variables, prediction_horizon + 1>  x_states; // + 1 x0 is added 
	Matrix<double, num_manipulated_variables, mpc_control_horizon> deriv_wrt_u;
	Matrix<double, num_manipulated_variables, 1>   u, u_past;
	double residuum, residuum_signal, residuum_state;

	x_states.setZero();
	deriv_wrt_u.setZero();
	u.setZero();
	u_past.setZero();

	lambdas_x.setZero();
	lambdas_u.setZero();
	lambdas_u_ref.setZero();


	deriv_wrt_u.setZero();
	x_states.block(0, 0, x0_.rows(), x0_.cols()) = x0_;

	//u_past.block(0,0,u_current.rows(), u_current.cols())= u_current;
	u_past = 0 * u_current; //  MatrixXd::Zero(B_.cols(), 1);

   //bilo je horizon -1
	for (int i = 0; i < pred_horizon; i++) {
		if (i < control_horizon) {
			u << x[0][0 * control_horizon + (i)], x[0][0 * control_horizon + (i)],
				x[0][1 * control_horizon + (i)], -x[0][1 * control_horizon + (i)];

		}
		x_states.block(0, i + 1, x0_.rows(), x0_.cols()) = (A_ * x_states.block(0, i, x0_.rows(), x0_.cols()) + B_ * u).eval();
		lambdas_x.block(0, i, x0_.rows(), x0_.cols()) = -1 * x_ss_ + x_states.block(0, i, x0_.rows(), x0_.cols());



		lambdas_u.block(0, i, u_past.rows(), u_past.cols()) = u - u_past;
		lambdas_u_ref.block(0, i, u.rows(), u.cols()) = u - u_ss_;


		//derivation of u
		
		switch (i) {
		case 0:
			deriv_wrt_u.block(0, i, u.rows(), u.cols()) = ((2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past));
			break;
		case 1: case 2: case 3: case 4:
			deriv_wrt_u.block(0, i, u.rows(), u.cols()) = ((2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past));
			deriv_wrt_u.block(0, i - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, i - 1, u.rows(), u.cols()) - 2 * R_delta_*u).eval();
			break;
		default:
			deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) + (2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past)).eval();


			deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) - 2 * R_delta_*u).eval();
			break;

		}



		//derivation of x
		for (int j = 0; j <= i; j++) {


			switch (j)
			{
			case 0: case 1: case 2: case 3: case 4:
				deriv_wrt_u.block(0, j, u.rows(), u.cols()) = (deriv_wrt_u.block(0, j, u.rows(), u.cols()) + ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose()).eval();
				break;

			default:
				deriv_wrt_u.block(0, 4, u.rows(), u.cols()) = (deriv_wrt_u.block(0, 4, u.rows(), u.cols()) + ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose()).eval();
				break;
			}


		}

		u_past = u;
	}

	lambdas_u_ref = scale_MV_inv * lambdas_u_ref;
	lambdas_u = scale_MV_inv * lambdas_u;
	lambdas_x = scale_OV_inv * lambdas_x;

    x_states_1  =  x_states;


	residuum_signal = (lambdas_u_ref.cwiseProduct(R_*lambdas_u_ref)).sum() + (lambdas_u.cwiseProduct(R_delta_*lambdas_u)).sum();

	residuum_state = (lambdas_x.cwiseProduct(Q_*lambdas_x)).sum();

	residuum = residuum_signal + residuum_state;

	//cout << " residuum "<<residuum << "  "<<isnan(residuum) << endl;
	if (isnan(residuum)) residuum = 9e50;


	//ROS_INFO_STREAM("u_ss   : "<< u_ss_);
	//ROS_INFO_STREAM("x_ss_   : "<< x_ss_);
	//ROS_INFO_STREAM("deriv_wrt_u   : "<< deriv_wrt_u);
	residuals[0] = residuum;

	if (jacobians != NULL) {
		if (jacobians[0] != NULL) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < control_horizon; j++) {

					jacobians[0][i*control_horizon + j] = 2 * deriv_wrt_u(i * 2, j);
					
				}
			}
		}

	}

	return true;
}


