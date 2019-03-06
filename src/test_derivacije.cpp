#include <vector>
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

int main(int argc, char **argv){
    const int kStateSize = 8;
    const int kInputSize = 4;
    const int kMeasurementSize = 1;
    
     MatrixXd  u_ss_, x_ss_, x0_, u_prev_, x_states, u,  deriv_wrt_u, u_past, lambdas_x, lambdas_u,lambdas_u_ref, u_horizon, u_current, scale_MV_inv, scale_OV_inv, A_pow_matrix;
    
    double residuum, residuum_signal, residuum_state;
    Eigen::Matrix<double, kStateSize, 1> target_state;
    Eigen::Matrix<double, kStateSize, 1> current_state;
    Eigen::Matrix<double, kInputSize, 1> target_input;
    Eigen::Matrix<double, kStateSize, kStateSize>       model_A_, model_A_70_ms, A_;   
    Eigen::Matrix<double, kStateSize, kInputSize>       model_B_, model_B_70_ms, B_;   

    Eigen::Matrix<double, kInputSize, 1> r_command_;

    Eigen::VectorXd ref(kMeasurementSize);
    Eigen::Matrix<double, kStateSize, kStateSize>       A_contin;   
    Eigen::Matrix<double, kStateSize, kInputSize>       B_contin;
    Eigen::Matrix<double, kStateSize, kStateSize>       OV_scale;
    Eigen::Matrix<double, kInputSize, kInputSize>       MV_scale;
    Eigen::Matrix<double, kStateSize, kStateSize> Q, Q_;
    Eigen::Matrix<double, kStateSize, kStateSize> Q_final, Q_final_;
    Eigen::MatrixXd R, R_;
    Eigen::MatrixXd R_delta, R_delta_;

                    
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
    Q(6,6) = 2.7068000;
    Q(7,7) =0.676700;

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
                    
              
                int pred_horizon = 14;
                int control_horizon = 5; 
                 A_ = model_A_70_ms;
                 B_ = model_B_70_ms; 
                 
                  Q_ = Q; 
                  Q_final_ = Q_final;
                  R_= R;
                  R_delta_ = R_delta;
                    


                
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
                scale_MV_inv = MV_scale.inverse();
                scale_OV_inv = OV_scale.inverse();
                A_pow_matrix = MatrixXd::Identity(A_.rows(), A_.cols());
                
                 
 deriv_wrt_u.setZero();       
        x_states.block(0,0,x0_.rows(), x0_.cols()) = x0_;
        
        //u_past.block(0,0,u_current.rows(), u_current.cols())= u_current;

       //bilo je horizon -1
        for (int i = 0; i< pred_horizon; i++){
            if(i < control_horizon){
            u << 0,0,0,0;

            }
            x_states.block(0,i+1,x0_.rows(), x0_.cols()) = A_ * x_states.block(0,i,x0_.rows(), x0_.cols()) + B_*u; // + insecure_;
            lambdas_x.block(0,i,x0_.rows(), x0_.cols())  = -1*x_ss_ + x_states.block(0,i,x0_.rows(), x0_.cols());
            lambdas_u.block(0,i,u.rows(), u.cols()) = u - u_past;
            lambdas_u_ref.block(0,i,u.rows(), u.cols()) = u - u_ss_;
            //derivation of u
            if(i < control_horizon){
                deriv_wrt_u.block(0,i,u.rows(), u.cols()) +=(2*R_*u) -2*R_*u_ss_  + (4*R_delta_*u)+ (-2*R_delta_*u_past);
            
                if(i > 0){ deriv_wrt_u.block(0,i-1,u.rows(), u.cols()) +=(deriv_wrt_u.block(0,i-1,u.rows(), u.cols()) -2*R_delta_*u).eval();}
            }else{
                deriv_wrt_u.block(0,4,u.rows(), u.cols()) += (2*R_*u) -2*R_*u_ss_  + (4*R_delta_*u)+ (-2*R_delta_*u_past);
            
                deriv_wrt_u.block(0,4,u.rows(), u.cols()) += (deriv_wrt_u.block(0,4,u.rows(), u.cols()) -2*R_delta_*u).eval();
                
            }
            //derivation of x
            for(int j = 0; j<= i; j++){
                A_pow_matrix.setIdentity();
                 
                for(int k = 0; k <= i-j; k++){
                    A_pow_matrix = (A_pow_matrix*A_).eval();
                }
                if(j < control_horizon && i < control_horizon){
          
                
                deriv_wrt_u.block(0,j,u.rows(), u.cols()) +=((2*Q_*x_states.block(0,i+1,x0_.rows(), x0_.cols()) - 2*Q_*x_ss_).transpose()*(A_pow_matrix)*B_).transpose();
            
                    
                }else{
                    if(j >= control_horizon){
                    deriv_wrt_u.block(0,4,u.rows(), u.cols()) += ((2*Q_*x_states.block(0,i+1,x0_.rows(), x0_.cols()) - 2*Q_*x_ss_).transpose()*(A_pow_matrix)*B_).transpose();
                
                    }else{
                       deriv_wrt_u.block(0,j,u.rows(), u.cols()) +=((2*Q_*x_states.block(0,i+1,x0_.rows(), x0_.cols()) - 2*Q_*x_ss_).transpose()*(A_pow_matrix)*B_).transpose();
                    }                    
                    
                }
            
            }
            
            u_past.block(0,0,u.rows(), u.cols()) = u;
        }
        //cout << "jakobijan " << deriv_wrt_u << endl;
        lambdas_u_ref =scale_MV_inv * lambdas_u_ref;
        lambdas_u = scale_MV_inv * lambdas_u;
        lambdas_x = scale_OV_inv * lambdas_x;
        
        residuum_signal=(lambdas_u_ref.cwiseProduct(R_*lambdas_u_ref)).sum()  + (lambdas_u.cwiseProduct(R_delta_*lambdas_u)).sum();
         
         residuum_state=(lambdas_x.cwiseProduct(Q_*lambdas_x)).sum() ;
         // +((-x_ss_ + x_states.block(0,horizon,x0_.rows(),x0_.cols())).transpose()*Q_final_*(-x_ss_ + x_states.block(0,horizon,x0_.rows(),x0_.cols()))).sum() );
        
        residuum =  residuum_signal +  residuum_state ;
        
}
