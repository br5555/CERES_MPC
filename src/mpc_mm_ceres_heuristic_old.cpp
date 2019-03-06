#include <morus_control/mpc_mm_ceres.h>

MPC_cost::MPC_cost( MatrixXd A, MatrixXd B, MatrixXd Bd, MatrixXd Q,
                       MatrixXd Q_final, MatrixXd R, MatrixXd R_delta,
                        MatrixXd disturbance , int num_params, int pred_horizon,int control_horizon ,MatrixXd scale_MV,MatrixXd scale_OV) : num_params_(num_params),pred_horizon(pred_horizon), control_horizon(control_horizon) ,A_(A), B_(B), Bd_(Bd), Q_(Q), Q_final_(Q_final), R_(R),R_delta_(R_delta), disturbance_(disturbance)
            {   
                this->insecure_ = this->Bd_ * disturbance;
                x_states = MatrixXd::Zero(A_.rows(), pred_horizon+1);
                deriv_wrt_u =  MatrixXd::Zero(B_.cols(), control_horizon);
                u = MatrixXd::Zero(B_.cols(), 1);
                u_past = MatrixXd::Zero(B_.cols(), 1);
                u_current = MatrixXd::Zero(B_.cols(), 1);
                u_ss_ = MatrixXd::Zero(B_.cols(), 1);
                lambdas_x = MatrixXd::Zero(A_.rows(), pred_horizon);
                lambdas_u = MatrixXd::Zero(B_.cols(), pred_horizon);
                lambdas_u_ref = MatrixXd::Zero(B_.cols(), pred_horizon);
                u_horizon = MatrixXd::Zero(B_.cols(), pred_horizon);
                lambdas_u_ref = MatrixXd::Zero(B_.cols(), pred_horizon);
                x0_ =  MatrixXd::Zero(x_states.rows(), 1);
                x_ss_ =  MatrixXd::Zero(x_states.rows(), 1);
                scale_MV_inv = scale_MV.inverse();
                scale_OV_inv = scale_OV.inverse();
                A_pow_matrix = MatrixXd::Identity(A_.rows(), A_.cols());
                A_pow_B_cache = MatrixXd::Zero(A_.rows(), pred_horizon*B_.cols());
               A_pow_B_cache.block(0,0,A_.rows(), B_.cols()) = MatrixXd::Identity(A_.rows(), A_.cols())* B_;
               //cout << "pow " << 0 <<endl;
                 //   cout << A_pow_B_cache.block(0,(0),A_.rows(), B_.cols()) <<endl;
                for(int i = 0; i < pred_horizon-1; i++){
                
                    A_pow_B_cache.block(0, (i + 1)*B_.cols(), A_.rows(), B_.cols()) = (A_* A_pow_B_cache.block(0, (i)*B_.cols(), A_.rows(), B_.cols())).eval();
                   // cout << "pow " << i+1 << endl;
                    //cout << A_pow_B_cache.block(0,(i+1),A_.rows(), B_.cols()) <<endl;
                }
                cout << endl;
                //cout << A_pow_B_cache << endl;
                
            }

//When you add the const keyword to a method the this pointer will essentially become a pointer to const object, and you cannot therefore change any member data. (This is not completely true, because you can mark a member as mutable and a const method can then change it. It's mostly used for internal counters and stuff.).

bool MPC_cost::Evaluate(double const* const* x,
                                   double* residuals,
                                   double** jacobians) const {
 



        deriv_wrt_u.setZero();       
        x_states.block(0,0,x0_.rows(), x0_.cols()) = x0_;
        
        //u_past.block(0,0,u_current.rows(), u_current.cols())= u_current;
		u_past  = 0*u_current; //  MatrixXd::Zero(B_.cols(), 1);

       //bilo je horizon -1
        for (int i = 0; i< pred_horizon; i++){
           if(i < control_horizon){
            u << x[0][0*control_horizon + (i)], x[0][0*control_horizon+(i)],           
                 x[0][1*control_horizon + (i)], -x[0][1*control_horizon + (i)];

            }
            x_states.block(0,i+1,x0_.rows(), x0_.cols()) = A_ * x_states.block(0,i,x0_.rows(), x0_.cols()) + B_*u + insecure_;
           lambdas_x.block(0,i,x0_.rows(), x0_.cols())  = -1*x_ss_ + x_states.block(0,i,x0_.rows(), x0_.cols());
           
           
            lambdas_u.block(0,i,u_past.rows(), u_past.cols()) = u - u_past;
            lambdas_u_ref.block(0,i,u.rows(), u.cols()) = u - u_ss_;
            //derivation of u
            if (i < control_horizon) {
				deriv_wrt_u.block(0, i, u.rows(), u.cols()) = (deriv_wrt_u.block(0, i, u.rows(), u.cols()) + (2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past)).eval();
				//cout << "Derivacije prvi dio" << endl;
				//cout << deriv_wrt_u << endl;
				//cout << "Dodajem u prvi dio " << endl << (2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past) << endl;

				if (i > 0) {
					deriv_wrt_u.block(0, i - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, i - 1, u.rows(), u.cols()) - 2 * R_delta_*u).eval();
					//cout << "Derivacije drugi dio" << endl;
					//cout << deriv_wrt_u << endl;
					//cout << "Dodajem u drugi dio " << endl << (- 2 * R_delta_*u) << endl;
				}
			}
			else {
				deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) + (2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past)).eval();
				//cout << "Derivacije treci dio" << endl;
				//cout << deriv_wrt_u << endl;
				//cout << "Dodajem u treci dio " << endl << (2 * R_*u) - 2 * R_*u_ss_ + (4 * R_delta_*u) + (-2 * R_delta_*u_past) << endl;

				deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) = (deriv_wrt_u.block(0, control_horizon - 1, u.rows(), u.cols()) - 2 * R_delta_*u).eval();
				//cout << "Derivacije cetvrti dio" << endl;
				//cout << deriv_wrt_u << endl;
				//cout << "Dodajem u cetvrti dio " << endl << (deriv_wrt_u.block(0, 4, u.rows(), u.cols()) - 2 * R_delta_*u) << endl;

			}
			//derivation of x
			for (int j = 0; j <= i; j++) {


				if (j < control_horizon && i < control_horizon) {

					//bilo je promjenjen indeks u x_states_block
					//deriv_wrt_u.block(0, j, u.rows(), u.cols()) = (deriv_wrt_u.block(0, j, u.rows(), u.cols()) + ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j), A_.rows(), B_.cols())).transpose()).eval();
					deriv_wrt_u.block(0, j, u.rows(), u.cols()) = (deriv_wrt_u.block(0, j, u.rows(), u.cols()) + ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose()).eval();
					/*//cout << i - j << endl;
					//cout << "Chache" << A_pow_B_cache.block(0, (i - j), A_.rows(), B_.cols()) << endl;*/
					//cout << "Derivacije peti dio" << endl;
					//cout << deriv_wrt_u << endl;
					//cout << "Dodajem u peti dio " << endl << ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose() << endl;
					//cout << "Peti dio prva zagrada" << endl << (2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()<< endl;
					//cout << "Peti dio prva zagrada" << endl << A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols()) << endl;
					//cout  << "x_states" << endl << x_states << endl;
					//cout << "samo dio" << endl;
					//cout << x_states.block(0, i + 1, x0_.rows(), x0_.cols()) << endl;
					//cout << "Samo provjera" << endl;
					//cout << x_ss_ << endl;
				}
				else {
					if (j >= control_horizon) {
						deriv_wrt_u.block(0, 4, u.rows(), u.cols()) = (deriv_wrt_u.block(0, 4, u.rows(), u.cols()) + ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose()).eval();
						//cout << "Derivacije seti dio" << endl;
						//cout << deriv_wrt_u << endl;
						//cout << "Dodajem u sesti dio " << endl << ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose() << endl;
						
					}
					else {
						deriv_wrt_u.block(0, j, u.rows(), u.cols()) = (deriv_wrt_u.block(0, j, u.rows(), u.cols()) + ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose()).eval();
						//cout << "Derivacije sedmi dio" << endl;
						//cout << deriv_wrt_u << endl;
						//cout << "Dodajem u sedmi dio " << endl << ((2 * Q_*x_states.block(0, i + 1, x0_.rows(), x0_.cols()) - 2 * Q_*x_ss_).transpose()*A_pow_B_cache.block(0, (i - j)*B_.cols(), A_.rows(), B_.cols())).transpose() << endl;

					}

				}

			}

			//u_past.block(0,0,u_past.rows(), u_past.cols()) = u;
			u_past = u;
			//cout << "Derivacije" << endl;
			//cout << deriv_wrt_u << endl;
		}
        //cout << "jakobijan " << deriv_wrt_u << endl;
        lambdas_u_ref =scale_MV_inv * lambdas_u_ref;
        lambdas_u = scale_MV_inv * lambdas_u;
        lambdas_x = scale_OV_inv * lambdas_x;
        
        residuum_signal=(lambdas_u_ref.cwiseProduct(R_*lambdas_u_ref)).sum()  + (lambdas_u.cwiseProduct(R_delta_*lambdas_u)).sum();
         
         residuum_state=(lambdas_x.cwiseProduct(Q_*lambdas_x)).sum() ;
         // +((-x_ss_ + x_states.block(0,horizon,x0_.rows(),x0_.cols())).transpose()*Q_final_*(-x_ss_ + x_states.block(0,horizon,x0_.rows(),x0_.cols()))).sum() );
        
        residuum =  residuum_signal +  residuum_state ;//+ 1e1*(abs((lambdas_u_ref.block(0,0,1, pred_horizon) - lambdas_u_ref.block(1,0,1, pred_horizon)).sum())) +      1e1*(abs((lambdas_u_ref.block(2,0,1, pred_horizon) + lambdas_u_ref.block(3,0,1, pred_horizon)).sum()))             ;

        //cout << " residuum "<<residuum << "  "<<isnan(residuum) << endl;
        if(isnan(residuum)) residuum = 9e50;
        
        
        //ROS_INFO_STREAM("u_ss   : "<< u_ss_);
        //ROS_INFO_STREAM("x_ss_   : "<< x_ss_);
        //ROS_INFO_STREAM("deriv_wrt_u   : "<< deriv_wrt_u);
        residuals[0] = residuum;

        if (jacobians != NULL) {
                if (jacobians[0] != NULL) {
                    for(int i = 0; i<2; i++){
                        for(int j = 0; j< control_horizon; j++){
                            
                            jacobians[0][i*control_horizon + j] = 2*deriv_wrt_u(i*2,j);
                            //ROS_INFO_STREAM("jakobijan   : "<< jacobians[0][i*control_horizon + j]);
                            }
                        } 
                }
            
        }

        return true;
}

 
