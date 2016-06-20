#include <opencv2\opencv.hpp>
#include <stdlib.h>
#include <iomanip>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const double EYES_AVG = 65;
const double DEYE_AVG = 33;
const double M_AVG = 66;
const double FACE_AVG = 0.61812297734627819;

// 46
// 65
// 31

ofstream myfile;

CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;
CascadeClassifier smilesCascade;

void findFace(string name)
{
	Mat frame = imread(name);

	Mat image;
	cvtColor(frame, image, CV_BGR2GRAY);

	equalizeHist(image, image);

	vector<Rect> faces;
	vector<Rect> rectangles;
	faceCascade.detectMultiScale(image, faces, 1.1, 3);

	vector<vector<Rect>> faceEyes;
	vector<vector<Rect>> faceMouth;

	Mat output;
	cvtColor(image, output, CV_GRAY2BGR);

	int countEyes = 0;
	double totalEyes = 0;

	int countDEye = 0;
	double totalDEye = 0;

	int countMouth = 0;
	double totalMouth = 0;

	double ratio = 0;

	for (int i = 0; i < faces.size(); ++i) {
		// Transformar quadrado para retangulos
		double wDiff = faces[i].width - (0.764 * faces[i].width);
		double hDiff = faces[i].height - (1.236 * faces[i].height);

		double w = 0.764 * faces[i].width;
		double h = 1.236 * faces[i].height;

		double x = faces[i].x + (wDiff / 2);
		double y = faces[i].y + (hDiff / 2);
		
		ratio = w / h;
		
		// Desenha o quadrado e o retangulo
		rectangle(output, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), Scalar(0, 255, 255));
		rectangle(output, Rect(x, y, w, h), Scalar(0, 0, 255));

		rectangles.push_back(Rect(x, y, w, h));

		// Calcula os olhos
		Mat faceROI(image, faces[i]);
		vector<Rect> eyes;
		eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 10);

		faceEyes.push_back(eyes);

		// Desenha os olhos encontrados e guarda as posições
		for (int j = 0; j < eyes.size(); ++j)
		{
			countEyes++;
			totalEyes += eyes[0].y + (eyes[0].height / 2);

			rectangle(output, Rect(eyes[j].tl() + faces[i].tl(), eyes[j].size()), Scalar(0, 255, 0));
		}
	}

	if (true) {
		// Média das posições dos olhos
		double eyeAvg = EYES_AVG; // totalEyes / countEyes;

		for (int i = 0; i < faces.size(); ++i)
		{
			int dEye = rectangles[i].width / 1.865;
			int yEye = eyeAvg;

			// Desenha a linha dos olhos para faces que não tem nenhum olho
			// Usa apenas a média para calcular a posição
			if (faceEyes[i].size() == 0) 
			{
				rectangle(output, Rect(rectangles[i].x + (rectangles[i].width / 2) - dEye / 2, faces[i].y + eyeAvg, dEye, 1), Scalar(255, 255, 0));

				//cout << "Y: " << faces[i].y + eyeAvg << "\n";
			}
			else
			{
				// Caso tenha algum olho, calcula a posição baseada no olho encontrado
				yEye = faceEyes[i][0].y + (faceEyes[i][0].height / 2);

				// Se o olho está muito distante da média, usa a média
				//if (abs(eyeAvg - yEye) > EYES_AVG / 10) // 3
				//{
				//	yEye = eyeAvg;
					//rectangle(output, Rect(rectangles[i].x + (rectangles[i].width / 2) - dEye / 2, faces[i].y + eyeAvg, dEye, 1), Scalar(255, 0, 0));
				//}
				//else
				{
					// Caso contrário, usa a posição baseada no olho
					rectangle(output, Rect(rectangles[i].x + (rectangles[i].width / 2) - dEye / 2, faces[i].y + yEye, dEye, 1), Scalar(255, 0, 255));
				}

				//cout << "Y: " << faces[i].y + yEye << "\n";
			}

			/*myfile << "#LX LY	RX RY" << "\n";
			myfile << (rectangles[i].x + (rectangles[i].width / 2) - dEye / 2) + dEye << " ";
			myfile << faces[i].y + yEye << " ";
			myfile << rectangles[i].x + (rectangles[i].width / 2) - dEye / 2 << " ";
			myfile << faces[i].y + yEye << "\n";*/

			//cout << "X: " << rectangles[i].x + (rectangles[i].width / 2) - dEye / 2 << "\n";
			//cout << "dEye: " << dEye << "\n";

			countDEye++;
			totalDEye += dEye;

			// Calcula os sorrisos
			Mat faceROI(image, faces[i]);
			vector<Rect> smile;
			vector<int> numDetection;

			smilesCascade.detectMultiScale(faceROI, smile, numDetection, 1.1, 10);

			for (int j = 0; j < smile.size(); ++j)
			{
				// Calcula a proporção do tamanho da boca
				double m = dEye / 1.618;

				// se a boca estiver numa posição correta
				if (smile[j].y - yEye > 30) // 15
				{
					// Salva os valores para calcular a média
					countMouth++;
					totalMouth += smile[j].y + smile[j].height / 2;

					// Desenha
					rectangle(output, Rect(faces[i].x + faces[i].width / 2 - m / 2, faces[i].y + smile[j].y + smile[j].height / 2, m, 1), Scalar(255, 0, 0));
					vector<Rect> empty;
					faceMouth.push_back(empty);
				}
				else
					faceMouth.push_back(smile); // caso contrário, salva
			}
		}
	}

	if (true)
	{
		// Calcula a média do tamanho da boca
		double mAvg = M_AVG; // totalMouth / countMouth;
		// Calcula a média da distância dos olhos
		double dEyeAvg = DEYE_AVG; // totalDEye / countDEye;

		// Calcula a proporção do tamanho da boca
		double m = dEyeAvg / 1.618;

		// Desenha a distância da boca para as faces restantes
		for (int i = 0; i < faces.size(); ++i)
		{
			if (faceMouth.size() > i && faceMouth[i].size() > 0) 
			{
				rectangle(output, Rect(faces[i].x + faces[i].width / 2 - m / 2, faces[i].y + mAvg, m, 1), Scalar(0, 0, 255));
			}
		}
	}

	 imshow("Image", output);
}

void findFace2(string name)
{
	Mat frame = imread(name);

	Mat image;
	cvtColor(frame, image, CV_BGR2GRAY);

	equalizeHist(image, image);

	vector<Rect> faces;
	faceCascade.detectMultiScale(image, faces, 1.1, 3);

	Mat output;
	cvtColor(image, output, CV_GRAY2BGR);
	
	for (int i = 0; i < faces.size(); ++i) {
		// Desenha o quadrado e o retangulo
		rectangle(output, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), Scalar(0, 255, 255));

		// Calcula os olhos
		Mat faceROI(image, faces[i]);
		vector<Rect> eyes;
		eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 10);

		vector<int> x;
		vector<int> y;

		// Desenha os olhos encontrados e guarda as posições
		for (int j = 0; j < eyes.size(); ++j)
		{
			rectangle(output, Rect(eyes[j].tl() + faces[i].tl(), eyes[j].size()), Scalar(0, 255, 0));

			x.push_back(eyes[j].x + faces[i].x + eyes[j].width / 2);
			y.push_back(eyes[j].y + faces[i].y + eyes[j].height / 2);
		}

		if (x.size() > 0)
			myfile << "#LX LY	RX RY" << "\n";

		if (x.size() == 0) 
		{
			
		}
		else if (x.size() == 1) 
		{
			myfile << x[0] << " ";
			myfile << y[0] << " ";
			myfile << x[0] << " ";
			myfile << y[0] << "\r";
		}
		else if (x.size() > 1)
		{
			if (x[0] > x[1])
			{
				myfile << x[0] << " ";
				myfile << y[0] << " ";
				myfile << x[1] << " ";
				myfile << y[1] << "\r";
			}
			else
			{
				myfile << x[1] << " ";
				myfile << y[1] << " ";
				myfile << x[0] << " ";
				myfile << y[0] << "\r";
			}
		}
	}

	//imshow("Image", output);
}

int image = 0;

void on_trackbar(int, void*)
{
	cout << "Imagem: " << image << "\n";

	stringstream name;

	name << "data/BioID-FaceDatabase-V1.2/BioID_" << setfill('0') << setw(4) << image << ".pgm";

	findFace(name.str());
}

int main()
{
	faceCascade.load("haarcascade_frontalface_default.xml");
	eyesCascade.load("haarcascade_eye.xml");
	smilesCascade.load("haarcascade_smile.xml");

	/*	findFace2("data/BioID-FaceDatabase-V1.2/BioID_0001.pgm");
	
	createTrackbar("Choose image", "Image", &image, 30, on_trackbar);

	/// Show some stuff
	on_trackbar(image, 0);*/

	//findFace("data/BioID-FaceDatabase-V1.2/BioID_0002.pgm");
	//findFace("data/all.png");
	/*findFace("data/002_2.jpg");
	findFace("data/002_3.jpg");
	findFace("data/002_4.jpg");*/

	myfile.open("bioID3.txt");
	for (int i = 0; i < 1521; i++)
	{
		try {
			myfile << "Image: " << i << "\n";

			stringstream name;
			name << "data/BioID-FaceDatabase-V1.2/BioID_" << setfill('0') << setw(4) << i << ".pgm";
			findFace2(name.str());

			myfile << "\n";
		}
		catch (...) 
		{
			myfile << "Image: " << i << " error" << "\n";
		}
	}
	myfile.close();

	waitKey();

	return 0;
}