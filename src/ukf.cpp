#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <stdlib.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  //set state dimension
  n_x_ = 5;

  // set measurement state dimension
  n_z_ = 3;

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0; // we were given 30, but from video guidance max should be 6 and sigma should be half that

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3; //(sigma rho)

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_x_ + 1);

  //set augmented dimension
  n_aug_ = 7;

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  //set vector for weights
  weights_ = VectorXd(2*n_aug_+1);
   double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  NIS_laser_ = 0.103; // two degrees of freedom, 95% chi-sq value
  NIS_radar_ = 0.352; // three degrees of freedom, 95% chi-sq value
  is_initialized_= false;

  time_us_ = 0; // TODO, this is a WAG, need to revisit it

  S_  = MatrixXd(n_z_,n_z_);


}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
	  if (!is_initialized_) {
	    /**
	      * Initialize the state x_ with the first measurement.
	      * Create the covariance matrix.
	      * Remember: you'll need to convert radar from polar to cartesian coordinates.
	    */
		// first measurement
	    cout << "Initialize x_: " << endl;
	    for (int i=0; i< n_x_ ;i++)
	    	x_(i) = 1;

	    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
	      /**
	      Convert radar from polar to cartesian coordinates and initialize state.
	      */

	    }
	    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
	      /**
	      Initialize state.
	      */
	    }
	    // if x and y values are 0, set to 0.1
		if ( abs((double)x_[0]) < 0.1 && abs((double)x_[1]) < 0.1)
		{
			x_[0] = 0.1;
			x_[1] = 0.1;

		}
	    // done initializing, no need to predict or update
	    is_initialized_ = true;
	    return;
	  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();
  //calculate sigma points ...
  MatrixXd X0 = x_;
  MatrixXd X1 = MatrixXd(n_x_, n_x_);
  MatrixXd X2 = MatrixXd(n_x_, n_x_);
  for(int i=0; i < n_x_; i++){
      X1.col(i) = x_ + sqrt(lambda_ + n_x_)* A.col(i);
      X2.col(i) = x_ - sqrt(lambda_ + n_x_)* A.col(i);
  }

  //set sigma points as columns of matrix Xsig
  Xsig_pred_ << X0, X1, X2;

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_x_ , n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  double delta_t2 =pow(delta_t, 2);

  //predict sigma points
  for (int i = 0; i< 2*n_aug_ +1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t2 * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t2 * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t2;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_+ weights_(i) * Xsig_pred.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_+ weights_(i) * x_diff * x_diff.transpose() ;
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
	z_ = VectorXd(3);
	z_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1],
			meas_package.raw_measurements_[2];
	  //create matrix for cross correlation Tc
	  MatrixXd Tc = MatrixXd(n_x_, n_z_);
	  //create matrix for sigma points in measurement space
	  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);


	  //calculate cross correlation matrix
	  Tc.fill(0.0);
	  for (int i = 0; i < 2*n_aug_ + 1 ; i++) {

		// residual
		VectorXd z_diff = Zsig.col(i) - z_pred_;
	    //angle normalization
	    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

		// state difference
	    VectorXd x_diff = Xsig_pred_.col(i) - x_;

	    //angle normalization
	    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
	    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

	    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	  }
	  //calculate Kalman gain K;
	  MatrixXd K = Tc*S_.inverse();

	  //residual
	  VectorXd z_diff = z_ - z_pred_;

	  //angle normalization
	  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

	  //update state mean and covariance matrix
	  x_ = x_ + K*z_diff;
	  P_ = P_ - K * S_ * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
	  //create matrix for sigma points in measurement space
	  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);
	  //transform sigma points into measurement space
	  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

	    // extract values for better readability
	    double p_x = Xsig_pred_(0,i);
	    double p_y = Xsig_pred_(1,i);
	    double v  = Xsig_pred_(2,i);
	    double yaw = Xsig_pred_(3,i);

	    double v1 = cos(yaw)*v;
	    double v2 = sin(yaw)*v;

	    // measurement model
	    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
	    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
	    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
	  }

	  z_pred_.fill(0.0);
	  for (int i=0; i < 2*n_aug_ +1; i++) {
	      z_pred_ = z_pred_ + weights_(i) * Zsig.col(i);
	  }

	  //measurement covariance matrix S
	  MatrixXd S = MatrixXd(n_z_,n_z_);
	  S.fill(0.0);
	  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
	    //residual
	    VectorXd z_diff = Zsig.col(i) - z_pred_;

	    //angle normalization
	    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

	    S = S + weights_(i) * z_diff * z_diff.transpose();
	  }

	  //add measurement noise covariance matrix
	  MatrixXd R = MatrixXd(n_z_,n_z_);
	  R <<    std_radr_*std_radr_, 0, 0,
	          0, std_radphi_*std_radphi_, 0,
	          0, 0,std_radrd_*std_radrd_;
	  S = S + R;
}
