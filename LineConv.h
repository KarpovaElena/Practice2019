#pragma once// реализация гомографии
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


typedef struct __retdata {
	std::vector<cv::Point> dots;
	cv::Point masscenter;
} retdata;

typedef struct __retdataf {
	cv::Point ln[2];
	cv::Point masscenter;
	std::vector<cv::Point> dots;
	bool isnew;
	cv::Point2f trackedpoint;
} retdataf;

class LineConv {
protected:
	
	//this float gives the paramets of find data area
	
	float xn1, yn, xn2;
	float xh1, yh, xh2;

	//arcLparam - param is perimeter of rectanle
	//if rectangle perimeter is less then this param it puts in trash
	
	float arcLparam;
	
	float hxn1, hyn, hxn2;
	float hxh1, hyh, hxh2;
	bool _hpadded;
	void createhomogrdet(cv::Mat& img);
	//function applies homograph transform by add homograph params
	void _delbadarea(cv::Mat& img);                           //select area of interests
	void createbritn(cv::Mat& img, double gamma = 2.2);      // gamma correction
	void rdatamake(cv::Mat& img);                            
	std::vector<retdata> _rdta;
public:
	LineConv(float txn1, float txn2, float tyn, float txh1, float txh2, float tyh, float tarcLparam);
	LineConv() = delete;
	void addhomograph(float txn1, float txn2, float tyn, float txh1, float txh2, float tyh);
	bool ishomographactive();
	virtual void bind(cv::Mat& image); // image will be change in the action, using reference for better perfomance;
	std::vector<retdata> getdata(); 
};

typedef struct __ticrecogmat {
	cv::Mat limg;
	std::vector<cv::Point2f> bdots;
	int tic;
}_trm;

class AdjLineConv : public LineConv {
	//Function makes line of the line, makes it straight by the course
	std::vector<cv::Point> dirline(std::vector<cv::Point> b);
	std::vector<retdataf> _rdta;
	void rdatamakef(cv::Mat& img);
	//making treaking object by optical flow
	void fndprew(cv::Mat& img);
	float dotrange; //shows max radius acception dot out of contour;
	int tics; 
	std::vector<cv::Point2f> lvec;
	std::vector<_trm> ttm;
	cv::Mat _prew;
	bool beg;
public:
	AdjLineConv(float txn1, float txn2, float tyn, float txh1, float txh2, float tyh, float tarcLparam, float fpr, int tic = 3);
	AdjLineConv() = delete;
	virtual void bind(cv::Mat& image);
	std::vector<retdataf> getdata();
};
