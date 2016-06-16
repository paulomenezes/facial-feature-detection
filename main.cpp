#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
 	//VideoCapture capture(0);

	//while (true)
	//{
		Mat frame = imread("Lenna.png");

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
		faceCascade.detectMultiScale(image, faces);

		Mat output;
		cvtColor(image, output, CV_GRAY2BGR);

		for (int i = 0; i < faces.size(); ++i) {
			rectangle(output, faces[i], Scalar(0, 0, 255));

			Mat faceROI(image, faces[i]);
			vector<Rect> eyes;
			vector<int> numDetection;
			eyesCascade.detectMultiScale(faceROI, eyes, numDetection, 1.1, 10);

			for (int j = 0; j < eyes.size(); ++j)
			{
				rectangle(output, Rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, eyes[j].width, eyes[j].height), Scalar(0, 10 * numDetection[j], 0));
				//rectangle(output, Rect(eyes[j].tl() + faces[i].tl(), eyes[j].size()), Scalar(0, 10 * numDetection[j], 0));
				cout << eyes[j].size() << "\n";
			}

			vector<Rect> smile;
			vector<int> numDetection2;
			smilesCascade.detectMultiScale(faceROI, smile, numDetection2, 1.1, 10);

			for (int j = 0; j < smile.size(); ++j)
			{
				rectangle(output, Rect(smile[j].tl() + faces[i].tl(), smile[j].size()), Scalar(255, 0, 0));
			}
	//	}

		imshow("faces", output);
		waitKey();
	}
	
	return 0;
}
