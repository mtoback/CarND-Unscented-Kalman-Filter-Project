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
  P_ <<     1, 0, 0,   0,    0,
		  0, 1, 0,   0,    0,
		  0, 0, 1000.0,0,    0,
		  0, 0, 0,   1000.0, 0,
		  0, 0, 0,   0,    1000.0;
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.5; // we were given 30, but from video guidance max should be much smaller

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;// given 30 but again should be much smaller;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3; //(sigma rho)

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.0175;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.1;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  //set augmented dimension
  n_aug_ = 7;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_ = MatrixXd(n_x_, 2 * n_x_ + 1);
  z_pred_ = VectorXd(n_z_);

  //define spreading parameter
  lambda_ = 3 - n_z_;

  //set vector for weights
  InitializeWeights();

  NIS_laser_ = 0.103; // two degrees of freedom, 95% chi-sq value
  NIS_radar_ = 0.352; // three degrees of freedom, 95% chi-sq value
  // set zeroth value when starting
  is_initialized_= false;

  // need to use basic sqrt calculation on first datapoint
  is_first_datapoint_ = true;

  time_us_ = 0; // previous timestamp

  S_  = MatrixXd(n_z_,n_z_);


}

UKF::~UKF() {}

/**
 * define weights as specified in ACC02 paper
 */
void UKF::InitializeWeights(){
	//set vector for weights
	weights_ = VectorXd(2*n_aug_+1);
	weights_(0) = lambda_/(lambda_+n_aug_);
	weights_(1) = (1 - weights_(0))/(pow(2,n_aug_));
	weights_(2) = weights_(1);
	for (int i=3; i< 2*n_aug_+1; i++) {
		weights_(i) = weights_(1)*pow(2, i-1);
	}

}
void UKF::InitializeMeasurement(MeasurementPackage meas_package){
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
	  double rho = meas_package.raw_measurements_[0];
	  double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];
	  x_ << rho*cos(phi), rho*sin(phi), 0.0, 0.0, 0.0;
	  time_us_ = meas_package.timestamp_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
		//set the state with the initial location and zero velocity
		x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0.0, 0.0, 0.0;

		time_us_ = meas_package.timestamp_;
    }
    is_first_datapoint_ = true;
    P_ << 1, 0, 0,   0,    0,
  		  0, 1, 0,   0,    0,
  		  0, 0, 2,0,    0,
  		  0, 0, 0,   2, 0,
  		  0, 0, 0,   0,    2;

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
	  if (!is_initialized_) {
		InitializeMeasurement(meas_package);
		  is_initialized_ = true;
		  return;
	  }

	    // if x and y values are 0, set to 0.1
		if ( abs((double)x_[0]) < 0.1 && abs((double)x_[1]) < 0.1)
		{
			x_[0] = 0.1;
			x_[1] = 0.1;

		}
		double delta_t = (meas_package.timestamp_ - time_us_)/ 1000000.0;	//dt - expressed in seconds
		time_us_ = meas_package.timestamp_;
		/**
		 * as described in prediction, when Cholesky decomposition fails,
		 * restart the filter using the previous measurement as a state initializer
		 */
		try {
			while(delta_t > 0.1){
				const double dt = 0.05;
				Prediction(dt);
				delta_t -= dt;
			}
			Prediction(delta_t);
		} catch(std::range_error e){
			// If convariance matrix is non positive definite (because of numerical instability),
			// restart the filter using previous measurement as initialiser.
			InitializeMeasurement(previous_measurement_);
			// Redo prediction using the current measurement
			// We don't get exception this time, because initial P (identity) is positive definite.
			delta_t = (meas_package.timestamp_ - previous_measurement_.timestamp_)/ 1000000.0;
			while(delta_t > 0.1){
				const double dt = 0.05;
				Prediction(dt);
				delta_t -= dt;
			}
			Prediction(delta_t);
		}
		if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			UpdateLidar(meas_package);
		} else {
			UpdateRadar(meas_package);
		}
	}

void UKF::GenerateSigmaPoints(){
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
  Xsig_ << X0, X1, X2;
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
  GenerateSigmaPoints();
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

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
  /*
   * I got stuck with very large values for rho dot, creating an near infinite loop as the value goes to infinity.
   * From the forums:
   * Many have noticed, that with some (big enough) std_a and std_yawdd combinations, the implementation
   * given in the lessons gives Nan or inf results. This happens especially with the data set 2,
   * where Δt is big (1.0s). It also happens easily with the data set 1, if one forces the Δt time
   *  to 1.0s. When looking at the covariance matrices, one can see that values of the elements
   *  suddenly start to grow rapidly, and then end up to NaN or inf.
   *
   *  The other indication of the problem is that the computation seems to be stuck in an infinite loop at some point.
   *  This is because it tries to normalise very big angles in loops:
   *  while (a > M_PI) a -= 2.*M_PI;
   *  while (a < -M_PI) a += 2.*M_PI;
   *  My first effort to tackle the problem was to do angle normalisation in all intermediate phases.
   *  That was merely mitigating the consequences not the problem itself. So I had to change the approach.
   *  The reason for growing values seems to be consecutive failures of the matrix square root (Cholesky
   *  decomposition) of the covariance matrix P. And the reason, why the decomposition fails, is that
   *  the covariance matrix is not positive semi-definite (matrix has negative eigenvalues).
   *  Several references, that I've looked, mention that P should always be positive semi-definite.
   *  So why do we end up to invalid matrices. I can think of three choices:
   *
   *  1. Buggy algorithm/model
   *  2. Buggy implementation
   *  3. Invalid data
   *
   *  I've been working with the option 2. There seems to be quite a lot very small element values,
   *  positive or negative, during the intermediate phases of the calculation. The negative values
   *  may be just because of rounding errors etc., and thus give faulty negative values later in the pipeline,
   *  resulting invalid covariance matrices.
   */
  /**
   * Currently my solution is to restart the filter when I bump into a decomposition failure. This is how to detect failure
   * (use instead of line MatrixXd L = P_aug.llt().matrixL();
   * unless it is the first datapoint
   */
   MatrixXd L;
   if (is_first_datapoint_) {
       L = P_aug.llt().matrixL();
       is_first_datapoint_ = false;
   } else {
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
	  /*
	   * When Cholesky decomposition fails, I don't try to continue, but throw an exception
	   * and restart the filter using the previous measurement as a state initalizer.
	   */
	  // 2. get the lower triangle
	  L = lltOfPaug.matrixL();
   }
  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  //predict sigma points
  double delta_t2 =pow(delta_t, 2);

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
    // avoid division by zero
    if(fabs(p_x) < 0.0001){
    	p_x = 0.0001;
    }
    if (fabs(p_y) < 0.0001){
    	p_y = 0.0001;
    }
    //predicted state values
    double px_p, py_p;

    // avoid too large a yaw or yaw rate
    if (yaw > M_PI) {
    	yaw = M_PI;
    } else if (yaw < - M_PI){
    	yaw = -M_PI;
    }
    if (yawd > M_PI) {
    	yawd = M_PI;
    } else if (yawd < - M_PI){
    	yawd = -M_PI;
    }
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
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;

    // avoid division by zero
    if(fabs(px_p) < 0.0001){
    	px_p = 0.0001;
    }
    if (fabs(py_p) < 0.0001){
    	py_p = 0.0001;
    }


  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_+ weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  cout << "x values: " << x_ << endl;
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);
    cout << "Sigma point prediction in col " << i << ":" << Xsig_pred_.col(i) << endl;
    //angle normalization
    cout << "angle:" << x_diff(3) << endl;
    while (x_diff(3)> M_PI){
    	x_diff(3)-=2.*M_PI;
    	cout << "angle:" << x_diff(3) << endl;
    }
    while (x_diff(3)<-M_PI){
    	x_diff(3)+=2.*M_PI;
    	cout << "angle:" << x_diff(3) << endl;
    }

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
	  //create matrix for cross correlation Tc
	  MatrixXd Tc = MatrixXd(n_x_, n_z_);
	  // create a vector to hold the change in measurements
	  VectorXd z_diff = VectorXd(n_z_);
	  //create matrix for sigma points in measurement space
	  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);
	  //transform sigma points into measurement space
	  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

		    // extract values for better readability
		    double p_x = Xsig_pred_(0,i);
		    double p_y = Xsig_pred_(1,i);
		    double v  = Xsig_pred_(2,i);
		    double yaw = Xsig_pred_(3,i);
		    // catch initial lidar of zero
		    if((fabs(p_x) < 0.005)){
		    	p_x = 0.005;
		    }
		    if((fabs(p_y) < 0.005)){
		    	p_y = 0.005;
		    }
		    // can't have a negative velocity
		    if (v<0)
		    	v = -v;
		    double v1 = cos(yaw)*v;
		    double v2 = sin(yaw)*v;

		    // measurement model, handling case were position is 0 separately
		    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
		    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
		    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
	  }

	  // Mean predicted measurement
	  z_pred_.fill(0.0);
	  for (int i=0; i < 2*n_aug_ +1; i++) {
	      z_pred_ = z_pred_ + weights_(i) * Zsig.col(i);
	  }
	  // calculate measurement covariance matrix S
	  S_.fill(0.0);
	  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
	    //residual
	    VectorXd z_diff = Zsig.col(i) - z_pred_;

	    //angle normalization
	    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

	    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
	  }

	  //add measurement noise covariance matrix
	  MatrixXd R = MatrixXd(n_z_,n_z_);
	  R <<    std_radr_*std_radr_, 0, 0,
	          0, std_radphi_*std_radphi_, 0,
	          0, 0,std_radrd_*std_radrd_;
	  S_ = S_ + R;

	  //calculate cross correlation matrix
	  Tc.fill(0.0);
	  for (int i = 0; i < 2*n_aug_ + 1 ; i++) {

		// residual
		z_diff = Zsig.col(i) - z_pred_;
	    //angle normalization
	    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

		// state difference
	    VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);

	    //angle normalization
	    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
	    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

	    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	  }
	  //calculate Kalman gain K;
	  MatrixXd K = Tc*S_.inverse();

	  //angle normalization
	  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

	  //update state mean and covariance matrix
	  x_ = x_ + K*z_diff;
	  P_ = P_ - K * S_ * K.transpose();


	  NIS_laser_ = z_diff.transpose()*S_.inverse()*z_diff;
	  cout << "NIS for lidar = " << NIS_laser_ << endl;
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

	    // avoid division by zero
	    if(fabs(p_x) < 0.0001){
	    	p_x = 0.0001;
	    }
	    if (fabs(p_y) < 0.0001){
	    	p_y = 0.0001;
	    }
	    // measurement model
	    Zsig(0,i) = meas_package.raw_measurements_[0];  //r
	    Zsig(1,i) = meas_package.raw_measurements_[1];  //phi
	    Zsig(2,i) = meas_package.raw_measurements_[0];  //r_dot
	  }

	  z_pred_.fill(0.0);
	  for (int i=0; i < 2*n_aug_ +1; i++) {
	      z_pred_ = z_pred_ + weights_(i) * Zsig.col(i);
	  }

	  //measurement covariance matrix S
	  S_.fill(0.0);
	  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
	    //residual
	    VectorXd z_diff = Zsig.col(i) - z_pred_;

	    //angle normalization
	    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
	    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

	    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
		// state difference
	    VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);

	    //angle normalization
	    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
	    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
	    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
	    //add measurement noise covariance matrix
	    MatrixXd R = MatrixXd(n_z_,n_z_);
		R <<    std_radr_*std_radr_, 0, 0,
		          0, std_radphi_*std_radphi_, 0,
		          0, 0,std_radrd_*std_radrd_;
		S_ = S_ + R;
	    NIS_radar_ = z_diff.transpose()*S_.inverse()*z_diff;
		cout << "NIS for radar = " << NIS_radar_ << endl;
	  }

}
