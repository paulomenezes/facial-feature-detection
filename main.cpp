#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
	Mat image = imread("Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);
	
	equalizeHist(image, image);
	
	CascadeClassifier faceCascade;
	
	faceCascade.load("haarcascade_frontalface_default.xml");
	
	vector<Rect> faces;
	faceCascade.detectMultiScale(image, faces);
	
	Mat output;
	cvtColor(image, output, CV_GRAY2BGR);
	
	for (int i = 0; i < faces.size(); ++i) {
		rectangle(output, faces[i], Scalar(0, 0, 255));
	}
	
	imshow("faces", output);
	waitKey();
	
	return 0;
}
