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

  // initialize covariance matrix to large values, positive definite
  P_ = MatrixXd(n_x_, n_x_);
  P_ <<     0.5, 0, 0,   0,    0,
		  0, 0.5, 0,   0,    0,
		  0, 0, 0.25,0,    0,
		  0, 0, 0,   0.25, 0,
		  0, 0, 0,   0,    0.25;
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3.0;

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
  NIS_laser_ = 0.0;
  NIS_radar_ = 0.0;
  is_initialized_ = false;
  //define spreading parameter

  n_aug_ = 7;
  n_x_ = 5;
  lambda_ = 3 - n_x_;

  time_us_ = 0;

  is_first_datapoint_ = true;

}

UKF::~UKF() {}

void UKF::InitializeMeasurement(MeasurementPackage meas_package){
    /**
    TODO:
      * Initialize the state x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
	// first measurement
    cout << "UKF: " << endl;
    x_ = VectorXd(n_x_);
    x_ << 1, 1, 0, 0 ,0;
    P_ = MatrixXd(n_x_, n_x_);
    P_ <<  0.5, 0, 0,   0,    0,
  		  0, 0.5, 0,   0,    0,
  		  0, 0, 0.25,0,    0,
  		  0, 0, 0,   0.25, 0,
  		  0, 0, 0,   0,    0.25;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
    	  VectorXd z = VectorXd(3);
    	  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1],
    			  meas_package.raw_measurements_[2];
		  x_ << z[0]*cos((double)z[1]), z[0]*sin((double)z[1]), 0.0, 0.0, 0.0;
		  time_us_  = meas_package.timestamp_;
		  previous_measurement_ = meas_package;
		  cout << "initial x_ = " << endl << x_ << endl;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
		//set the state with the initial location and zero velocity
		x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0;

		time_us_ = meas_package.timestamp_;
		previous_measurement_ = meas_package;
		cout << "initial x_ = " << endl << x_ << endl;
    }
    // if x and y values are near 0, set to 0.01
	if ( fabs(x_[0]) < 0.01 && fabs(x_[1]) < 0.01)
	{
		x_[0] = 0.01;
		x_[1] = 0.01;

	}
    // done initializing, no need to predict or update
	cout << "initial P=" <<endl << P_ << endl;
    is_initialized_ = true;
    return;
  }

MatrixXd UKF::GenerateSigmaPoints() {


  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  //set first column of sigma point matrix
  Xsig.col(0)  = x_;

  //set remaining sigma points
  for (int i = 0; i < n_x_; i++)
  {
    Xsig.col(i+1)     = x_ + sqrt(lambda_+n_x_) * A.col(i);
    Xsig.col(i+1+n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }

  //print result
  std::cout << "Xsig = " << std::endl << Xsig << std::endl;

  //write result
  return Xsig;
}

MatrixXd UKF::AugmentedSigmaPoints() {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);
  x_aug.fill(0.0);
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  P_aug.fill(0.0);
  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  // Take matrix square root
  // 1. compute the Cholesky decomposition of P_aug
  Eigen::LLT<MatrixXd> lltOfPaug(P_aug);
  if (lltOfPaug.info() == Eigen::NumericalIssue) {
      // if decomposition fails, we have numerical issues
      std::cout << "LLT failed!" << std::endl;
      //Eigen::EigenSolver<MatrixXd> es(P_aug);
      //cout << "Eigenvalues of P_aug:" << endl << es.eigenvalues() << endl;
      throw std::range_error("LLT failed");
  }
  // 2. get the lower triangle
  MatrixXd L = lltOfPaug.matrixL();
  double sqrt_lambda_n_aug = sqrt(lambda_+n_aug_);
  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug +  sqrt_lambda_n_aug* L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt_lambda_n_aug * L.col(i);
  }

  //write result
  return Xsig_aug;


}
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
  cout << "==================================================" << endl;
  cout << "measurement = " << meas_package.raw_measurements_ << endl;
  cout << "timestamp= " << meas_package.timestamp_  << endl;
  if(is_initialized_){
	  MatrixXd Xsig_pred;
	  double delta_t = (meas_package.timestamp_ - time_us_)/1000000.0;
	  cout << "delta t = " << delta_t << endl;
	  try {
		  while (delta_t > 0.1)
		  {
			  const double dt = 0.05;
			  Xsig_pred = Prediction(dt);
			  delta_t -= dt;
		  }
		  Xsig_pred = Prediction(delta_t);
	  } catch(std::range_error e){
		    // If convariance matrix is non positive definite (because of numerical instability),
		    // restart the filter using previous measurement as initialiser.
		    InitializeMeasurement(previous_measurement_);
		    // Redo prediction using the current measurement
		    // We don't get exception this time, because initial P (identity) is positive definite.
		    Xsig_pred = Prediction(delta_t);
	  }
	  cout << "updated mean = " << x_ << endl;
	  cout << "updated covariance = " << P_ << endl;
      if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    	  UpdateRadar(meas_package, Xsig_pred);
    	  cout << "NIS_Radar = " << NIS_radar_ << endl;
      } else {
    	  UpdateLidar(meas_package, Xsig_pred);
    	  cout << "NIS_Lidar = " << NIS_laser_ << endl;
      }
      time_us_ = meas_package.timestamp_;
      previous_measurement_ = meas_package;

  } else {
     InitializeMeasurement(meas_package);
  }
}

MatrixXd UKF::SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t) {

  int n_z_ = 5;
  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_z_, 2 * n_aug_ + 1);


  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
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
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  //print result
  //std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  //write result
  return Xsig_pred;

}
void UKF::PredictMeanAndCovariance(MatrixXd Xsig_pred) {

  //create vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);


  // set weights
  double weight_0 = lambda_/(lambda_+ n_aug_);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_ + lambda_);
    weights(i) = weight;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_+ weights(i) * Xsig_pred.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - Xsig_pred.col(0);
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights(i) * x_diff * x_diff.transpose() ;
  }


  //print result
  //std::cout << "Predicted state" << std::endl;
  //std::cout << x_ << std::endl;
  //std::cout << "Predicted covariance matrix" << std::endl;
  //std::cout << P_ << std::endl;
}
/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
MatrixXd UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  MatrixXd x_augsig = AugmentedSigmaPoints();
  MatrixXd Xsig_pred = SigmaPointPrediction(x_augsig, delta_t);
  PredictMeanAndCovariance(Xsig_pred);
  cout << "after prediction" << endl;
  cout << "x=" <<endl << x_ << endl;
  cout << "p=" << endl << P_ << endl << endl;
  return Xsig_pred;
}

void UKF::PredictLidarMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
	MatrixXd R_laser_ = MatrixXd(2, 2);
	//measurement covariance matrix - laser
	R_laser_ << 0.15*0.15, 0,
	        0, 0.15*0.15;

	MatrixXd H_ = MatrixXd(2, 5);
	H_(0,0) = 1.0;
	H_(0,1) = 0.0;
	H_(0,2) = 0.0;
	H_(0,3) = 0.0;
	H_(0,4) = 0.0;
	H_(1,0) = 0.0;
	H_(1,1) = 1.0;
	H_(1,2) = 0.0;
	H_(1,3) = 0.0;
	H_(1,4) = 0.0;
	VectorXd y = meas_package.raw_measurements_ - H_*x_;

	MatrixXd Ht = H_.transpose();
	MatrixXd Si = (H_ * P_ * H_.transpose() + R_laser_).inverse(); // S.inverse();
	MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;

	VectorXd y_new = meas_package.raw_measurements_ - H_*x_;
	MatrixXd Si_new = (H_ * P_ * H_.transpose() + R_laser_).inverse();
	NIS_laser_ = y_new.transpose()*Si_new*y_new;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package, MatrixXd Xsig_pred) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  PredictLidarMeasurement(meas_package);
  cout << "after lidar update" << endl;
  cout << "x=" <<endl << x_ << endl;
  cout << "p=" << endl << P_ << endl << endl;
}

void UKF::UpdateState(MatrixXd Xsig_pred, MatrixXd Zsig, VectorXd z_pred,
		MatrixXd S, VectorXd z) {


  //set vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);
  weights.fill(0.0);
   double weight_0 = lambda_/(lambda_+n_aug_);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_ + lambda_);
    weights(i) = weight;
  }
  int n_z_ = 3;

  //create matrix for cross correlation Tc
  //(says nx x nz)
  MatrixXd Tc = MatrixXd(5, n_z_);


  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - Xsig_pred.col(0);
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  //angle normalization
  P_ = P_ - K*S*K.transpose();


  //print result
  //std::cout << "Updated state x: " << std::endl << x << std::endl;
  //std::cout << "Updated state covariance P: " << std::endl << P << std::endl;


}

void UKF::PredictRadarMeasurement(MatrixXd Xsig_pred, VectorXd* zpred, MatrixXd *Z_sig, MatrixXd* S_out) {


  //set vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);
   double weight_0 = lambda_/(lambda_+n_aug_);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {
    double weight = 0.5/(n_aug_+lambda_);
    weights(i) = weight;
  }


  int n_z_ = 3;
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readability
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);
    double v  = Xsig_pred(2,i);
    double yaw = Xsig_pred(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    if (p_x == 0 && p_y == 0){
    	Zsig(0,i) = 0;
    	Zsig(1,i) = 0;
    	Zsig(2,i) = 0;
    } else {
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S_ = MatrixXd(n_z_,n_z_);
  S_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S_ = S_ + weights(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_,n_z_);
  R <<    std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0,std_radrd_*std_radrd_;
  S_ = S_ + R;



  //print result
  //std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  //std::cout << "S: " << std::endl << S << std::endl;

  //write result
  *zpred = z_pred;
  *S_out = S_;
  *Z_sig = Zsig;
}
/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package, MatrixXd Xsig_pred) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z_ = 3;
  VectorXd z_pred = VectorXd(n_z_);
  VectorXd z_out = VectorXd(n_z_);
  MatrixXd S_out = MatrixXd(n_z_,n_z_);
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  PredictRadarMeasurement(Xsig_pred, &z_pred, &Zsig, &S_out);
  UpdateState(Xsig_pred, Zsig, z_pred, S_out, z_out);
  cout << "after rad update" << endl;
  cout << "x=" <<endl << x_ << endl;
  cout << "p=" << endl << P_ << endl << endl;

}
