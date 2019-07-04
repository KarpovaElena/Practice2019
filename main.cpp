//#include "Stickfind.h"
#include "LineConv.h"
#include <chrono>
#include <fstream>

using namespace cv;
using namespace std;


int main() {
	VideoCapture cap("C:\\Users\\Bear-\\OneDrive\\Рабочий стол\\Проекты\\Var1.3\\Road - 1894.mp4");
	if (!cap.isOpened()) 
		return -1;
	Mat im;
	AdjLineConv det(0.2, 0.8, 0.04, 0.4, 0.6, 0.25, 50, 20);
	for (; ;) {
		auto beg = std::chrono::high_resolution_clock::now();
		Mat frame;
		cap >> frame;
		Mat tk = frame.clone();
		if (frame.empty()) break;
		det.bind(frame);
		auto t = det.getdata();
		for (int i = 0; i < t.size(); i++) {
			line(tk, t[i].ln[0], t[i].ln[1], cv::Scalar(255, 0, 0), 2);
			if (t[i].isnew) putText(tk, "new_object", t[i].ln[0], 2, 1, cv::Scalar(0, 255, 0));
			else {
				putText(tk, "Tracked", t[i].ln[0], 2, 1, cv::Scalar(0, 0, 255));
				drawMarker(tk, t[i].trackedpoint, cv::Scalar(0, 0, 0), 2, 25);
			}
		}
		imshow("find", tk);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> fsec = end - beg;
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(fsec);
		int tme = ms.count();
		cout << tme << std::endl;
		if (tme < 30) tme = 30 - tme;
		else tme = 1;

		if (waitKey(tme) >= 0) break;
	}
	return 0;

}