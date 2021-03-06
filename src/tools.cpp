#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    if(estimations.size() != ground_truth.size()
       || estimations.size() == 0) {
        cout << "Estimations is either empty or differs from ground truth" << endl;
        return rmse;
    }

    //accumulate squared residuals
    VectorXd residuals(4);
    residuals << 0.0, 0.0, 0.0, 0.0;
    for(int i=0; i < estimations.size(); ++i){
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
        residuals += residual;
    }

    //calculate the mean
    residuals = residuals/estimations.size();

    //calculate the squared root
    rmse = residuals.array().sqrt();

    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

    MatrixXd Hj(3,4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    //pre-compute a set of terms to avoid repeated calculation
    float c1 = px * px + py * py;
    float c2 = sqrt(c1);
    float c3 = (c1 * c2);

    if (fabs(px) < 0.0001 and fabs(py) < 0.0001) {
        px = 0.0001;
        py = 0.0001;
    }
    //check division by zero
    if(fabs(c1) < 0.0001) {
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
        c1 = 0.0001;
    }

    //compute the Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0,
            -(py/c1), (px/c1), 0, 0,
            py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

    return Hj;
}
