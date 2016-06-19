#include <opencv2\opencv.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;

void findFace(string name)
{
	Mat frame = imread(name);

	Mat image;
	cvtColor(frame, image, CV_BGR2GRAY);

	equalizeHist(image, image);

	CascadeClassifier faceCascade;
	CascadeClassifier eyesCascade;
	CascadeClassifier smilesCascade;

	faceCascade.load("haarcascade_frontalface_default.xml");
	eyesCascade.load("haarcascade_eye.xml");
	smilesCascade.load("haarcascade_smile.xml");

	vector<Rect> faces;
	faceCascade.detectMultiScale(image, faces, 1.1, 10);

	vector<vector<Rect>> faceEyes;
	vector<vector<Rect>> faceMouth;

	Mat output;
	cvtColor(image, output, CV_GRAY2BGR);

	int countEyes = 0;
	int totalEyes = 0;

	int countDEye = 0;
	int totalDEye = 0;

	for (int i = 0; i < faces.size(); i++) {
		// Transformar quadrado para retangulos
		int wDiff = faces[i].width - (0.764 * faces[i].width);
		int hDiff = faces[i].height - (1.236 * faces[i].height);

		int w = 0.764 * faces[i].width;
		int h = 1.236 * faces[i].height;

		int x = faces[i].x + (wDiff / 2);
		int y = faces[i].y + (hDiff / 2);

		// Desenha o quadrado e o retangulo
		rectangle(output, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), Scalar(0, 255, 255));
		rectangle(output, Rect(x, y, w, h), Scalar(0, 0, 255));

		// Calcula os olhos
		Mat faceROI(image, faces[i]);
		vector<Rect> eyes;
		vector<int> numDetection;
		eyesCascade.detectMultiScale(faceROI, eyes, numDetection, 1.1);

		faceEyes.push_back(eyes);

		// Desenha os olhos encontrados e guarda as posições
		for (int j = 0; j < eyes.size(); j++)
		{
			countEyes++;
			totalEyes += eyes[0].y + (eyes[0].height / 2);

			rectangle(output, Rect(eyes[j].tl() + faces[i].tl(), eyes[j].size()), Scalar(0, 255, 0));
		}
	}

	int countMouth = 0;
	int totalMouth = 0;

	if (countEyes > 0) {
		// Média das posições dos olhos
		int eyeAvg = totalEyes / countEyes;

		for (int i = 0; i < faces.size(); i++)
		{
			int wDiff = faces[i].width - (0.764 * faces[i].width);
			int hDiff = faces[i].height - (1.236 * faces[i].height);

			int w = 0.764 * faces[i].width;
			int h = 1.236 * faces[i].height;

			int x = faces[i].x + (wDiff / 2);
			int y = faces[i].y + (hDiff / 2);

			int dEye = w / 1.865;
			int yEye = eyeAvg;

			// Desenha a linha dos olhos para faces que não tem nenhum olho
			// Usa apenas a média para calcular a posição
			if (faceEyes[i].size() == 0) 
			{
				rectangle(output, Rect(x + (w / 2) - dEye / 2, faces[i].y + eyeAvg, dEye, 1), Scalar(255, 255, 0));
			}
			else
			{
				// Caso tenha algum olho, calcula a posição baseada no olho encontrado
				yEye = faceEyes[i][0].y + (faceEyes[i][0].height / 2);

				// Se o olho está muito distante da média, usa a média
				if (abs(eyeAvg - yEye) > 3)
				{
					yEye = eyeAvg;
					rectangle(output, Rect(x + (w / 2) - dEye / 2, faces[i].y + yEye, dEye, 1), Scalar(255, 0, 0));
				}
				else
				{
					// Caso contrário, usa a posição baseada no olho
					rectangle(output, Rect(x + (w / 2) - dEye / 2, faces[i].y + yEye, dEye, 1), Scalar(255, 0, 255));
				}
			}

			countDEye++;
			totalDEye += dEye;

			// Calcula os sorrisos
			Mat faceROI(image, faces[i]);
			vector<Rect> smile;
			vector<int> numDetection2;
			smilesCascade.detectMultiScale(faceROI, smile, numDetection2, 1.1, 10);

			for (int j = 0; j < smile.size(); ++j)
			{
				// Calcula a proporção do tamanho da boca
				int m = dEye / 1.618;

				// se a boca não estiver numa posição incorreta
				if (smile[j].y - yEye > 15)
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

	if (countMouth > 0)
	{
		// Calcula a média do tamanho da boca
		int mAvg = totalMouth / countMouth;
		// Calcula a média da distância dos olhos
		int dEyeAvg = totalDEye / countDEye;

		// Calcula a proporção do tamanho da boca
		int m = dEyeAvg / 1.618;

		// Desenha a distância da boca para as faces restantes
		for (int i = 0; i < faces.size(); i++)
		{
			if (faceMouth[i].size() > 0) 
			{
				rectangle(output, Rect(faces[i].x + faces[i].width / 2 - m / 2, faces[i].y + mAvg, m, 1), Scalar(0, 0, 255));
			}
		}
	}

	imshow(name, output);
}

int main()
{
	//findFace("Lenna.png");
	findFace("data/all.png");
	/*findFace("data/002_2.jpg");
	findFace("data/002_3.jpg");
	findFace("data/002_4.jpg");*/

	waitKey();

	return 0;
}