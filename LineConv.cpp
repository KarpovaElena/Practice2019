#include "LineConv.h"

void LineConv::createhomogrdet(cv::Mat& img) {
	std::vector<cv::Point2f> baseP, makeP;
	int rows = img.rows;
	int cols = img.cols;
	baseP.push_back(cv::Point2f(cols * xn2, rows * yn));
	baseP.push_back(cv::Point2f(cols * xn1, rows * yn));
	baseP.push_back(cv::Point2f(cols * xh1, rows * yh));
	baseP.push_back(cv::Point2f(cols * xh2, rows * yh));

	makeP.push_back(cv::Point2f(cols * hxn2, rows * hyn));
	makeP.push_back(cv::Point2f(cols * hxn1, rows * hyn));
	makeP.push_back(cv::Point2f(cols * hxh1, rows * hyh));
	makeP.push_back(cv::Point2f(cols * hxh2, rows * hyh));

	cv::Mat t = cv::getPerspectiveTransform(baseP, makeP);
	cv::warpPerspective(img, img, t, img.size());
}

void LineConv::_delbadarea(cv::Mat& img) {
	int rows = img.rows;
	int cols = img.cols;
	cv::Mat im2(rows, cols, CV_8UC1, cv::Scalar(0));
	cv::Point pt[1][4];
	pt[0][0] = cv::Point(cols * xn2, rows * yn);// трапеция
	pt[0][1] = cv::Point(cols * xn1, rows * yn);
	pt[0][2] = cv::Point(cols * xh1, rows * yh);
	pt[0][3] = cv::Point(cols * xh2, rows * yh);
	const cv::Point* p[1] = { pt[0] };
	const int npt[] = { 4 };
	cv::fillPoly(im2, p, npt,
		1,
		cv::Scalar(255),
		8);
	cv::bitwise_and(img, im2, img);
}

void LineConv::createbritn(cv::Mat& img, double gamma) {// распознание границ трапеции
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	//cv::Mat res = img.clone();
	LUT(img, lookUpTable, img);
}


void LineConv::rdatamake(cv::Mat& img) {
	_rdta.clear();
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	for (int i = 0; i < contours.size(); i++) {
		retdata r;
		float g = cv::arcLength(cv::Mat(contours[i]), true);
		approxPolyDP(contours[i], r.dots, g * 0.03, true);
		if (r.dots.size() != 4 || g < arcLparam) continue;
		
		cv::Moments m = cv::moments(contours[i]);
		r.masscenter = cv::Point(m.m10 / m.m00, m.m01 / m.m00);
		_rdta.push_back(r);
	}
}

LineConv::LineConv(float txn1, float txn2, float tyn, float txh1, float txh2, float tyh, float tarcLparam) {
	xn1 = txn1;
	xn2 = txn2;
	xh1 = txh1;
	xh2 = txh2;
	yn = 1 - tyn;
	yh = 1 - tyh;
	arcLparam = tarcLparam;
	_hpadded = false;
}

void LineConv::addhomograph(float txn1, float txn2, float tyn, float txh1, float txh2, float tyh) {
	_hpadded = true;
	hxn1 = txn1;
	hxn2 = txn2;
	hxh1 = txh1;
	hxh2 = txh2;
	hyn = 1 - tyn;
	hyh = 1 - tyh;
}

bool LineConv::ishomographactive()
{
	return _hpadded;
}

void LineConv::bind(cv::Mat& image) {
	// forking... branch 1 ... creating GRAY image;
	cv::Mat im2 = image.clone();
	// correct gamma... make image darker
	createbritn(im2, 3);
	cv::cvtColor(im2, im2, cv::COLOR_RGB2GRAY);
	_delbadarea(im2);

	cv::inRange(im2, cv::Scalar(30), cv::Scalar(255), im2);
	
	
	cv::cvtColor(image, image, cv::COLOR_RGB2HLS);
	cv::inRange(image, cv::Scalar(0, 130, 0), cv::Scalar(220, 255, 255), image);
	
	cv::bitwise_and(image, im2, image);
	if (_hpadded) createhomogrdet(image);

	auto kn = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::morphologyEx(image, image, cv::MORPH_OPEN, kn);
	kn = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(image, image, cv::MORPH_CLOSE, kn);
	cv::Canny(image, image, 50, 250);
	rdatamake(image);
}

std::vector<retdata> LineConv::getdata()
{
	return _rdta;
}

std::vector<cv::Point> AdjLineConv::dirline(std::vector<cv::Point> b) {
	cv::Point2f _a, _b;
	_a.x = 0; _a.y = 0;
	_b.x = 0; _b.y = 0;
	std::vector<cv::Point> rv;
	bool t[4] = { 0, 0, 0, 0 };
	for (int j = 0; j < 2; j++) {
		int l = 0; float low = 0;
		for (int i = 0; i < 4; i++) {
			if (b[i].y > low && !t[i]) {
				l = i;
				low = b[i].y;
			}
		}
		t[l] = true;
	}
	for (int i = 0; i < 4; i++) {
		if (t[i]) {
			_a.x += b[i].x;
			_a.y += b[i].y;
		}
		else {
			_b.x += b[i].x;
			_b.y += b[i].y;
		}
	}
	_a = _a / 2; _b = _b / 2;
	rv.push_back(_a); rv.push_back(_b);
	return rv;
}

void AdjLineConv::rdatamakef(cv::Mat& img) {
	_rdta.clear();
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	std::vector<cv::Point2f> _lvec;
	for (int i = 0; i < contours.size(); i++) {
		retdataf f;
		f.isnew = true;
		float g = cv::arcLength(cv::Mat(contours[i]), true);
		approxPolyDP(contours[i], f.dots, g * 0.03, true);
		if (f.dots.size() != 4 || g < arcLparam) continue;
		auto kr = dirline(f.dots);
		f.ln[0] = kr[0]; f.ln[1] = kr[1];
		cv::Moments m = cv::moments(contours[i]);
		f.masscenter = cv::Point(m.m10 / m.m00, m.m01 / m.m00);
		bool fnd = false;
		for (int j = 0; j < lvec.size(); j++) {
			double k = cv::pointPolygonTest(contours[i], lvec[j], true);
			if (k<0 || k * (-1)>dotrange) {
				continue;
			}
			else {
				fnd = true;
				f.trackedpoint = lvec[j];
				break;
			}
		}
		if (fnd) f.isnew = false;
		_lvec.push_back(cv::Point2f(f.masscenter));
		_rdta.push_back(f);
	}
	if (_rdta.empty()) return;
	lvec = _lvec;
}

void AdjLineConv::fndprew(cv::Mat& img) {
	if (lvec.empty()) return;
	std::vector<cv::Point2f> pt, bp;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
	cv::calcOpticalFlowPyrLK(_prew, img, lvec, pt, status, err, cv::Size(15, 15), 2, criteria);
	lvec.clear();
	for (int i = 0; i < status.size(); i++) {
		if (status[i] == 1) {
			lvec.push_back(pt[i]);
		}
		else {
			bp.push_back(pt[i]);
		}
	}
	pt.clear();
	status.clear();
	for (int i = 0; i < ttm.size(); i++) {
		if (--ttm[i].tic == 0 || ttm[i].bdots.empty()) continue;
		std::vector<cv::Point2f> _bp;
		cv::calcOpticalFlowPyrLK(ttm[i].limg, img, ttm[i].bdots, pt, status, err, cv::Size(15, 15), 2, criteria);
		ttm[i].bdots.clear();
		for (int o = 0; o < status.size(); o++) {
			if (status[o] == 1) {
				lvec.push_back(pt[o]);
			}
			else {
				if (ttm[i].tic != 1) ttm[i].bdots.push_back(pt[o]);
			}
		}
	}
	if (!ttm.empty()) {
		ttm.erase(std::remove_if(ttm.begin(), ttm.end(), [](_trm const& x) -> bool { return (x.tic <= 0); }), ttm.end());
	}
	if (!bp.empty()) {
		_trm trm;
		trm.tic = tics;
		trm.bdots = bp;
		trm.limg = img.clone();
		ttm.push_back(trm);
	}
	std::cout << " cm ="<< ttm.size() << std::endl;
}

AdjLineConv::AdjLineConv(float txn1, float txn2, float tyn, float txh1, float txh2, float tyh, float tarcLparam, float fpr, int tic) :LineConv(txn1, txn2, tyn, txh1, txh2, tyh, tarcLparam) {
	dotrange = fpr;
	tics = tic;
}

void AdjLineConv::bind(cv::Mat& image) {
	// forking... branch 1 ... creating GRAY image;
	cv::Mat im2 = image.clone();
	// correct gamma... make image darker
	cv::cvtColor(im2, im2, cv::COLOR_RGB2GRAY);
	cv::equalizeHist(im2, im2);
	createbritn(im2, 3);

	_delbadarea(im2);
	cv::inRange(im2, cv::Scalar(15), cv::Scalar(255), im2);
	cv::imshow("trapeze", im2);
	
	cv::cvtColor(image, image, cv::COLOR_RGB2HLS);
	cv::inRange(image, cv::Scalar(0, 125, 0), cv::Scalar(220, 255, 255), image);
	cv::imshow("hls", image);
	
	cv::bitwise_and(image, im2, image);
	if (_hpadded) createhomogrdet(image);
	auto kn = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::morphologyEx(image, image, cv::MORPH_OPEN, kn);
	kn = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(image, image, cv::MORPH_CLOSE, kn);
	cv::Canny(image, image, 50, 250);
	if (!beg) {
		beg = true;
	}
	else {
		fndprew(image);
	}
	_prew = image;
	rdatamakef(image);
}

std::vector<retdataf> AdjLineConv::getdata()
{
	return _rdta;
}