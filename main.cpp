#include <opencv2\opencv.hpp>
#include <stdlib.h>
#include <iomanip>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const double EYES_AVG = 58.14; // 65;
const double DEYE_AVG = 33;
const double M_AVG = 66;
const double FACE_AVG = 0.61812297734627819;
const double MOUTH_AVG = 144.25;

// 46
// 65
// 31

ofstream myfile;

string names[5] { "HD", "HT", "HT2", "HTR", "HPF" };
// HD, HT, HT2, HTR, HPF
CascadeClassifier classifiers[5];
CascadeClassifier eyesCascade;
CascadeClassifier smilesCascade;

int countEye = 0;
float totalEye = 0;

void findFace(string name)
{
	int countMouth = 0;
	float totalMouth = 0;

	Mat frame = imread(name);

	Mat image;
	cvtColor(frame, image, CV_BGR2GRAY);

	equalizeHist(image, image);

	int mlx = 0;
	int mly = 0;
	int mrx = 0;
	int mry = 0;

	for (int l = 0; l < 5; l++)
	{
		vector<Rect> faces;
		vector<Rect> rectangles;

		classifiers[l].detectMultiScale(image, faces, 1.1, 3);

		vector<vector<Rect>> faceEyes;

		Mat output;
		cvtColor(image, output, CV_GRAY2BGR);

		vector<vector<Rect>> faceMouth;

		for (int i = 0; i < faces.size(); ++i) {
			// Transformar quadrado para retangulos
			double wDiff = faces[i].width - (0.764 * faces[i].width);
			double hDiff = faces[i].height - (1.236 * faces[i].height);

			double w = 0.764 * faces[i].width;
			double h = 1.236 * faces[i].height;

			double x = faces[i].x + (wDiff / 2);
			double y = faces[i].y + (hDiff / 2);

			// Desenha o quadrado e o retangulo
			rectangle(output, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), Scalar(0, 255, 255));
			rectangle(output, Rect(x, y, w, h), Scalar(0, 0, 255));

			rectangles.push_back(Rect(x, y, w, h));

			// Calcula os olhos
			Mat faceROI(image, faces[i]);
			vector<Rect> eyes;
			eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 12);

			faceEyes.push_back(eyes);

			// Desenha os olhos encontrados e guarda as posições
			for (int j = 0; j < eyes.size(); ++j)
			{
				countEye++;
				totalEye += eyes[j].y + (eyes[j].height / 2);

				rectangle(output, Rect(eyes[j].tl() + faces[i].tl(), eyes[j].size()), Scalar(0, 255, 0));
			}
		}

		// Média das posições dos olhos
		double eyeAvg = EYES_AVG; // totalEyes / countEyes;

		if (faces.size() == 0)
		{
			float values[10] = { 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 };

			for (int k = 0; k < 10; k++)
			{
				myfile << names[l] << " -- " << values[k] << " : 0 0 0 0 \n";
			}
		}
		else
		{
			for (int i = 0; i < 1; ++i)
			{
				int dEye = rectangles[i].width / 1.865; // 2.425; //
				int yEye = eyeAvg;

				float values[10] = { 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 };

				for (int k = 0; k < 10; k++)
				{
					dEye = rectangles[i].width * values[k];

					// Desenha a linha dos olhos para faces que não tem nenhum olho
					// Usa apenas a média para calcular a posição
					if (faceEyes[i].size() == 0 || faceEyes[i].size() > 2)
					{
						rectangle(output, Rect(rectangles[i].x + (rectangles[i].width / 2) - dEye / 2, faces[i].y + yEye, dEye, 1), Scalar(255, 255, 0));
					}
					else
					{
						// Caso tenha algum olho, calcula a posição baseada no olho encontrado
						yEye = faceEyes[i][0].y + (faceEyes[i][0].height / 2);

						// Caso contrário, usa a posição baseada no olho
						rectangle(output, Rect(rectangles[i].x + (rectangles[i].width / 2) - dEye / 2, faces[i].y + yEye, dEye, 1), Scalar(255, 0, 255));
					}

					myfile << names[l] << " -- " << values[k] << " : ";
					myfile << (rectangles[i].x + (rectangles[i].width / 2) - dEye / 2) + dEye << " ";
					myfile << faces[i].y + yEye << " ";
					myfile << rectangles[i].x + (rectangles[i].width / 2) - dEye / 2 << " ";
					myfile << faces[i].y + yEye << "\n";
				}

				// Calcula os sorrisos
				Mat faceROI(image, faces[i]);
				vector<Rect> smile;
				vector<int> numDetection2;
				smilesCascade.detectMultiScale(faceROI, smile, numDetection2, 1.1, 10);

				for (int j = 0; j < smile.size(); ++j)
				{
					rectangle(output, Rect(smile[j].tl() + faces[i].tl(), smile[j].size()), Scalar(0, 0, 0));

					// Calcula a proporção do tamanho da boca
					int m = dEye / 1.618;

					// se a boca estiver numa posição correta
					if (smile[j].y - yEye > 50)
					{
						// Salva os valores para calcular a média
						countMouth++;
						totalMouth += smile[j].y + (float)smile[j].height / 2;

						// Desenha
						rectangle(output, Rect(faces[i].x + faces[i].width / 2 - m / 2, faces[i].y + smile[j].y + smile[j].height / 2, m, 1), Scalar(255, 255, 255));
						vector<Rect> empty;
						faceMouth.push_back(empty);

						mlx = (faces[i].x + faces[i].width / 2 - m / 2);
						mly = faces[i].y + smile[j].y + smile[j].height / 2;
						mrx = (faces[i].x + faces[i].width / 2 - m / 2) + m;
						mry = faces[i].y + smile[j].y + smile[j].height / 2;
					}
					else
						faceMouth.push_back(smile); // caso contrário, salva
				}
			}


			if (countMouth == 0)
			{
				// Calcula a média do tamanho da boca
				int mAvg = totalMouth / countMouth;

				// Desenha a distância da boca para as faces restantes
				int f = faces.size() > 1 ? 1 : faces.size();
				for (int i = 0; i < f; i++)
				{
					// Calcula a média da distância dos olhos
					int dEyeAvg = rectangles[i].width * 0.5;

					// Calcula a proporção do tamanho da boca
					int m = dEyeAvg / 1.618;

					//if (faceMouth.size() > 0) {
					//if (faceMouth[i].size() > 0)
					{
						rectangle(output, Rect(faces[i].x + faces[i].width / 2 - m / 2, faces[i].y + MOUTH_AVG, m, 1), Scalar(0, 0, 255));

						mlx = (faces[i].x + faces[i].width / 2 - m / 2);
						mly = faces[i].y + MOUTH_AVG;
						mrx = (faces[i].x + faces[i].width / 2 - m / 2) + m;
						mry = faces[i].y + MOUTH_AVG;
					}
					//}
				}
			}
		}

		//imshow("Image", output);
	}

	myfile << "MOUTH ";
	myfile << mlx << " ";
	myfile << mly << " ";
	myfile << mrx << " ";
	myfile << mry << "\n";
}

int image = 0;

void on_trackbar(int, void*)
{
	stringstream name;
	name << "data/BioID-FaceDatabase-V1.2/BioID_" << setfill('0') << setw(4) << image << ".pgm";

	findFace(name.str());
}

int main()
{
	classifiers[0].load("haarcascade_frontalface_default.xml");
	classifiers[1].load("haarcascade_frontalface_alt.xml");
	classifiers[2].load("haarcascade_frontalface_alt2.xml");
	classifiers[3].load("haarcascade_frontalface_alt_tree.xml");
	classifiers[4].load("haarcascade_profileface.xml");

	eyesCascade.load("haarcascade_eye.xml");
	smilesCascade.load("haarcascade_smile.xml");

	myfile.open("bioID8.txt");
	for (int i = 0; i < 1521; i++)
	{
		cout << i << "\n";
		stringstream name;
		name << "data/BioID-FaceDatabase-V1.2/BioID_" << setfill('0') << setw(4) << i << ".pgm";
		findFace(name.str());
	}
	myfile.close();

	float media = totalEye / countEye;
	//float media = totalMouth / countMouth;
	
	createTrackbar("Image", "Image", &image, 40, on_trackbar);

	waitKey();

	return 0;
}