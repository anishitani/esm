/*
 * ESM.cpp
 *
 *  Created on: Nov 23, 2013
 *      Author: nishitani
 */

/*
 * TODO Alteraçõe iniciadas, mas não terminadas.
 * Precisa terminar a alteração da forma de aquisição do tamanho da imagem.
 * Manter armazenada a última imagem de maneira que a cada iteração somente a
 * nova imagem seja passada.
 */

#include <esm.h>

void ESM::createGenerators() {
	A.clear();
	A = std::vector<cv::Mat>(dof);

	/*
	 * Movimento em X
	 */
	if (dof == 1) {
		A[0] =
				(cv::Mat_<float>(4, 4) << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	}

	/*
	 * Veículos com Instant Center of Rotation.
	 * A câmera deve estar perpendicular ao eixo
	 * do veículo
	 */
	if (dof == 2) {
		A[0] =
				(cv::Mat_<float>(4, 4) << 0, -1, 0, 0, 1, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0);
		A[1] =
				(cv::Mat_<float>(4, 4) << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	}

	/*
	 * Permite o movimento em x,y e rotação
	 * em torno de z.
	 */
	if (dof == 3) {
		A[0] =
				(cv::Mat_<float>(4, 4) << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		A[1] =
				(cv::Mat_<float>(4, 4) << 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0);
		A[2] =
				(cv::Mat_<float>(4, 4) << 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	}
}

/**
 * @todo Remove this method. Camera extrinsic should be singular.
 */
void ESM::pitchEffect() {
	cv::Mat Veh2Optical =
			(cv::Mat_<float>(4, 4) << 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1);

	/*
	 * Cria a matriz de rotação que tranforma coordenadas
	 * no sistema de coordenadas do carro para o sistema
	 * de coordenadas da câmera.
	 * Note que o considera-se que yaw e roll são nulos.
	 */
	if (extrinsic.empty()) {
		RotX =
				(cv::Mat_<float>(4, 4) << 1, 0, 0, 0, 0, cos(pitch), sin(pitch), 0, 0, -sin(
						pitch), cos(pitch), 0, 0, 0, 0, 1);
		this->setExtrinsic(cv::Mat(RotX * Veh2Optical));
	}
	this->planeNormal = this->extrinsic(cv::Rect(0, 0, 3, 3))
			* (cv::Mat_<float>(3, 1) << 0, 0, 1 / this->planeDistance);
	this->planeNormal.convertTo(this->planeNormal, this->matrixType);
}

bool ESM::process(cv::Mat TIRef, cv::Mat ICur, cv::Mat K, cv::Mat G0,
		cv::Mat &T) {
	float RMS;

	float bestRMS = 100000; // "Infinito"
	cv::Mat TOld, TNew;
	cv::Mat TBest;

	T.convertTo(TOld, this->matrixType);

	this->tempCols = TIRef.cols;
	this->tempRows = TIRef.rows;

	if (!prepare()) {
		fprintf(stderr, "Method's preparation failed.");
		return false;
	}

	for (int i = 0; i < this->maxIterations; i++) {

		TNew = TOld;
		bool converged = updateSSDSE2Motion(TIRef, ICur, this->roi.width,
				this->roi.height, K, G0, TNew, RMS);

		if (RMS < bestRMS) {
			bestRMS = RMS;
			TBest = TNew;
		} else {
			if(RMS<10.0)
				converged = true;
			else
				break;
		}

		if (converged) {
			T = TBest;
			RMS = bestRMS;

			return true;
		}
		TOld = TNew;
	}

	RMS = bestRMS;

	return false;
}

bool ESM::updateSSDSE2Motion(cv::Mat TIRef, cv::Mat ICur, float width,
		float height, cv::Mat K, cv::Mat G0, cv::Mat &T, float &RMS) {
	// K * V2C * T * inv(V2C) * nd * inv(K)
	cv::Mat _G = this->leftMatrix * T * this->rightMatrix;
	cv::Mat G(_G, cv::Rect(0, 0, 3, 3));

	cv::Mat mask;
	cv::Mat di(tempRows, tempCols, CV_32F, cv::Scalar(0));
	cv::Mat TICur;

	/*
	 * Warps the current image fitting the reference ROI.
	 * Points outside the image are set to -1.
	 */
	cv::warpPerspective(ICur, TICur, G * G0, cv::Size(width, height),
			cv::INTER_NEAREST + cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT,
			cv::Scalar(-1));

	di.release();
	cv::threshold(TICur, mask, 0, 255, cv::THRESH_BINARY);
	cv::subtract(TIRef, TICur, di, mask, CV_32F);

	// Root mean square error calculation
	cv::Mat di2;
	cv::pow(di, 2, di2);
	RMS = (float) cv::mean(di2, mask).val[0];
	RMS = std::sqrt(RMS);

	/* **********************************
	 * BEGIN TESTING
	 * **********************************/
	cv::Mat diff(TIRef.rows, 3 * TIRef.cols, CV_8U);

	cv::Rect win1(0 * TIRef.cols, 0, TIRef.cols, TIRef.rows);
	cv::Rect win2(1 * TIRef.cols, 0, TIRef.cols, TIRef.rows);
	cv::Rect win3(2 * TIRef.cols, 0, TIRef.cols, TIRef.rows);

	TIRef.copyTo(diff(win1));
	TICur.copyTo(diff(win2));
	di.copyTo(diff(win3));

//	char frase[256];
//	sprintf(frase, "RMS: %f", RMS);
//	cv::putText(diff, frase, cv::Point(3, 10), cv::FONT_HERSHEY_SIMPLEX, 0.3,
//			cv::Scalar(255));

	cv::imshow("diff", diff);

	clock_t t = clock();
	char f[256];
	sprintf(f, "/home/anishitani/Templates/%ld.png", t);
	std::cout << f << std::endl;
	cv::imwrite(f,diff);

	char key = cv::waitKey(10);
	if (key == 'q')
		exit(0);
	/* **********************************
	 * END TESTING
	 * **********************************/

	// Gradient of Reference and Current images
	cv::Mat dxRef, dyRef;
	cv::Mat dxCur, dyCur;

	// Gradient operation as defined in MATLAB
	gradient(TIRef, dxRef, dyRef);
	gradient(TICur, dxCur, dyCur);

	cv::Mat J = imgJacSE2planar(cv::Mat((dxRef + dxCur) / 2),
			cv::Mat((dyRef + dyCur) / 2), width, height, G0, T);

	cv::Mat Jinv = (J.t() * J).inv() * J.t();

	di = di.t();
	di = di.reshape(0, width * height);

	cv::Mat d = Jinv * di;

	if (cv::norm(d) < convCriteria)
		return 1;

	cv::Mat xA(4, 4, CV_32F, cv::Scalar(0));
	for (int i = 0; i < (int) A.size(); i++)
		xA += d.at<float>(i) * A[i];

	cv::Mat dT;
	Eigen::Matrix<float, 4, 4> _xA;
//	cv::cv2eigen(xA, _xA);
//	cv::eigen2cv(Eigen::Matrix<float, 4, 4>(_xA.exp()), dT);

	T = T * dT;

	return 0;
}

cv::Mat ESM::imgJacSE2planar(cv::Mat mIx, cv::Mat mIy, float width,
		float height, cv::Mat G0, cv::Mat &T) {
	cv::Mat Jesm;

	cv::Mat JTx(9, this->A.size(), CV_32F);
	for (int i = 0; i < (int) this->A.size(); i++) {
		cv::Mat JTxi = this->leftMatrix * this->A[i] * this->rightMatrix;
		JTxi(cv::Rect(0, 0, 3, 3)).clone().reshape(1, 9).copyTo(JTx.col(i));
	}

	cv::Mat dx = cv::Mat(mIx.t()).reshape(1, 1).t();
	cv::Mat dy = cv::Mat(mIy.t()).reshape(1, 1).t();

	cv::Mat px, py;
	meshgrid(G0.at<float>(0, 2), G0.at<float>(1, 2), width, height, px, py);
	px = cv::Mat(px.t()).reshape(1, 1).t();
	py = cv::Mat(py.t()).reshape(1, 1).t();

	cv::Mat JIW(width * height, 9, CV_32F);
	/*
	 * The first three equations are dx*p.
	 * The second triple are dy*p.
	 * The last triple are -(dx*px+dy*py)*p
	 */
	cv::multiply(dx, px, JIW.col(0));
	cv::multiply(dx, py, JIW.col(1));
	dx.copyTo(JIW.col(2));

	cv::multiply(dy, px, JIW.col(3));
	cv::multiply(dy, py, JIW.col(4));
	dy.copyTo(JIW.col(5));

	cv::multiply(-(JIW.col(0) + JIW.col(4)), px, JIW.col(6));
	cv::multiply(-(JIW.col(0) + JIW.col(4)), py, JIW.col(7));
	cv::Mat(-JIW.col(0) - JIW.col(4)).copyTo(JIW.col(8));

	Jesm = JIW * JTx;

	return Jesm;
}

bool ESM::prepare() {
	if (this->extrinsic.empty()) {
		fprintf(stderr, "Missing parameter: extrinsic matrix!\n");
		return false;
	}
	if (this->planeNormal.empty()) {
		fprintf(stderr, "Missing parameter: normal vector!\n");
		return false;
	}
	if (this->planeDistance < 10e-5) {
		fprintf(stderr, "Missing parameter: camera height!\n");
		return false;
	}

	cv::Mat nd = cv::Mat::zeros(4, 4, this->matrixType);
	cv::Mat(cv::Mat::eye(3, 3, this->matrixType)).copyTo(
			nd(cv::Rect(0, 0, 3, 3)));
	cv::Mat(-this->planeNormal.t() / this->planeDistance).copyTo(
			nd.row(3).colRange(0, 3));

	this->leftMatrix = this->intrinsic * this->extrinsic;
	this->rightMatrix = this->extrinsic.inv() * nd * this->intrinsicInv;

	return true;
}
