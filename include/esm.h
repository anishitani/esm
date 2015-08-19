/*
 * ESM.h
 *
 *  Created on: Sep 10, 2013
 *      Author: nishitani
 */

#ifndef ESM_H_
#define ESM_H_

#include <cstdio>
#include <vector>

#include <opencv2/opencv.hpp>

#include <math_tools.h>
#include <matrix_utils.h>

class ESM {
private:
	int maxIterations; ///< Número máximo de iterações para minimização
	float convCriteria; ///< Variação mínima para considerar a convergência

	float planeDistance; ///< Distance to the plane.
	int dof; ///< Graus de liberdade do movimento

	float pitch; ///< Inclinação da câmera
	int tempRows; ///< Número de linhas (altura) do template
	int tempCols; ///< Número de colunas (largura) do template

	cv::Mat TIRef; ///< Template (recorte) da imagem anterior
	cv::Mat ICur; ///< Imagem atual à qual o template será alinhado

	cv::Mat intrinsic; 		///< Parâmetros intrínsecos da câmera
	cv::Mat intrinsicInv; 	///< Parâmetros intrínsecos da câmera
	cv::Mat extrinsic; ///< Extrinsic parameters. Camera's pose wrt some reference system

	cv::Mat leftMatrix;		///< Matrix result of <Intrinsic> * <Extrinsic>
	cv::Mat rightMatrix;///< Matrix result of <Extrinsic>^-1 * <GroundPlaneNormal> * <Intrinsic>^-1

	cv::Mat planeNormal; ///< Vetor normal ao plano do chão
	cv::Rect roi; ///< Região de interesse da imagem;

	std::vector<cv::Mat> A; ///< Array das matrizes geradoras da algebra Lie

	cv::Mat RotX; ///< Rotação em torno do eixo-x (coordenadas do veículo)

	int matrixType; /// Matrix data type

	/**
	 * @todo Definir previamente os valores de:
	 * 	cv::Mat K;
	 * 	cv::Mat constMat1;
	 * 	cv::Mat constMat2;
	 * onde constMat1 = K*Veh2Cam e constMat2 = Veh2Cam.inv()*K.inv()
	 */

	/**
	 * @brief Cria os geradores da algebra lie.
	 */
	void createGenerators();

	/**
	 * @brief Realiza as alterações necessárias nas informações dependentes da inclinação da câmera
	 */
	void pitchEffect();

	/**
	 * @brief Passo de minimização para o movimento em SE(2)
	 *
	 * @param TIRef Template da imagem de referência (anterior)
	 * @param ICur Imagem atual
	 * @param width Largura da imagem
	 * @param height Altura da imagem
	 * @param K Matriz da câmera
	 * @param norVec Vetor normal ao plano do chão
	 * @param G0 Transformação aplicada ao template da imagem de referência para posicionamento inicial em relação à imagem atual
	 * @param T Transformação sendo atualizada na iteração
	 * @param RMS Erro ao final da iteração
	 * @return 1 em caso de sucesso
	 */
	bool updateSSDSE2Motion(cv::Mat TIRef, cv::Mat ICur, float width,
			float height, cv::Mat K, cv::Mat G0, cv::Mat &T, float &RMS);

	/**
	 * @brief Calculo do jacobiano para o caso de movimento planar
	 *
	 * @param mIx Média dos gradientes em X dos templates das imagens atual e anterior
	 * @param mIy Média dos gradientes em Y dos templates das imagens atual e anterior
	 * @param width Largura da imagem
	 * @param height Altura da imagem
	 * @param norVec Vetor normal ao plano do chão
	 * @param G0 Transformação aplicada ao template da imagem de referência para posicionamento inicial em relação à imagem atual
	 * @param T Transformação sendo atualizada na iteração
	 * @return Jacobiano
	 */
	cv::Mat imgJacSE2planar(cv::Mat mIx, cv::Mat mIy, float width, float height,
			cv::Mat G0, cv::Mat &T);

public:

	/**
	 * @brief Construtor da classe ESM. Pode ser inicializado com os valores de graus de liberdade e inclinação.
	 *
	 * @param camHeight Parâmetro obrigatório que define a altura da câmera e consequentemente a escala do movimento estimado
	 * @param dof Graus de liberdade do movimento
	 * @param pitch Inclinação da câmera
	 * @param prefix Posição do template na imagem
	 */
	ESM() {
		// Algorithm properties
		this->maxIterations = 50;
		this->convCriteria = 1.0e-2;
		this->dof = 3;

		// Matrix properties
		this->matrixType = CV_32F;

		createGenerators();
	}

	~ESM() {
	}

	bool process(cv::Mat TIRef, cv::Mat ICur, cv::Mat K, cv::Mat G0, cv::Mat &T);

	void setPlaneDistance(float planeDistance) {
		this->planeDistance = planeDistance;
	}

	void setPitch(float pitch) {
		this->pitch = pitch;
		pitchEffect();
	}

	/**
	 * Public method used to set the tracked plane normal wrt the camera.
	 * @param planeNormal Array with the plane normal.
	 */
	void setPlaneNormal(cv::InputArray planeNormal) {
		CV_Assert(planeNormal.type() == CV_32F || planeNormal.type() == CV_64F);

		cv::Mat m = planeNormal.getMat();
		int elems = m.rows * m.cols * m.channels();

		CV_Assert(elems == 3 || elems == 4);

		this->planeNormal = planeNormal.getMat().reshape(1, elems)(
				cv::Rect(0, 0, 1, 3));
	}

	void setDoF(float dof) {
		this->dof = dof;
		createGenerators();
	}

	void setMaxIterations(int maxIterations) {
		this->maxIterations = maxIterations;
	}

	void setConvergenceCriteria(float convCriteria) {
		this->convCriteria = convCriteria;
	}

	/**
	 * Public method for setting camera's intrinsic parameters and it's inverse.
	 * @param intrinsic Intrinsic parameters (3x3 Matrix).
	 */
	void setIntrinsic(cv::Mat intrinsic) {
		CV_Assert(intrinsic.type() == CV_32F || intrinsic.type() == CV_64F);

		// Matrix initialization
		this->intrinsic = cv::Mat::zeros(4, 4, this->matrixType);
		this->intrinsicInv = cv::Mat::zeros(4, 4, this->matrixType);

		intrinsic(cv::Rect(0, 0, 3, 3)).convertTo(
				this->intrinsic(cv::Rect(0, 0, 3, 3)), this->matrixType);
		this->intrinsicInv = invSubmatrix(this->intrinsic, 0, 0, 3, 3);
	}

	/**
	 * Public method used to get the camera's extrinsic parameters.
	 */
	cv::Mat getExtrinsic() {
		return this->extrinsic;
	}

	/**
	 * Public method used to set the camera's extrinsic parameters.
	 * @param extrinsic Camera's extrinsic parameters (pose wrt some reference).
	 */
	void setExtrinsic(cv::InputArray extrinsic) {
		CV_Assert(extrinsic.type() == CV_32F || extrinsic.type() == CV_64F);
		this->extrinsic = cv::Mat::eye(4, 4, this->matrixType);
		extrinsic.getMat()(cv::Rect(0, 0, 3, 3)).convertTo(
				this->extrinsic(cv::Rect(0, 0, 3, 3)), this->matrixType);
	}

	void setROI(cv::Rect roi) {
		this->roi = roi;
	}

	void setGenerators(std::vector<cv::Mat> A) {
		// Cria um grupo com os geradores fornecidos
		this->A.clear();
		this->A.assign(A.begin(), A.end());
	}

	cv::Mat getNormalVector() {
		return planeNormal;
	}

	/**
	 * Public method used to prepare the method environment (e.g. matrix pre-processing).
	 */
	bool prepare();
};

#endif /* ESM_H_ */

