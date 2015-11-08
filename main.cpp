#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat src, src_gray, croped_src, src_hsv;

int a = 10;
int b = 3;
int c = 10;

vector<Vec3f> detectBlueCircles(Mat src);
vector<Vec3f> detectRedCircles(Mat src);
vector<Vec3f> detectYellowCircles(Mat src);

Mat combineImages(vector<Mat> & images)
{
	int cols = 0;
	int rows = 0;
	int left = 0;
	for (int i = 0; i < images.size(); i++)
	{
		cols += images[i].cols;
		if (images[i].rows > rows)
			rows = images[i].rows;
	}
	Mat resultImage = Mat::zeros(Size(cols, rows), images[0].type());
	for (int i = 0; i < images.size(); i++)
	{
		images[i].copyTo(resultImage(Rect(left, 0, images[i].cols, images[i].rows)));
		left += images[i].cols;
	}
	return resultImage;
}

vector<vector<Vec3f>> extractTargets(Mat image)
{
	cvtColor(src, src_hsv, CV_BGR2HSV);
	vector<Mat> HSV;
	split(src_hsv, HSV);
	auto clahe = createCLAHE(3.0, Size(8, 8));
	clahe->apply(HSV[2], HSV[2]);
	merge(HSV, src_hsv);
	cvtColor(src_hsv, src, CV_HSV2BGR);
	//vector<Vec3f> circles = detectBlueCircles(src_hsv);
	vector<Vec3f> redCircles = detectRedCircles(src_hsv);
	vector<vector<Vec3f> > circles;
	for (int i = 10; i >= 6; i--)
	{
		circles.push_back(vector<Vec3f>());
		for (int j = 0; j < redCircles.size(); j++)
		{
			Vec3f circle = Vec3f(redCircles[j][0], redCircles[j][1], redCircles[j][2] * (10 - i + 1) / 4);
			circles.back().push_back(circle);
		}
	}
	return circles;
}

Mat cutTargets(Mat src, vector<vector<Vec3f> > targets)
{
	Mat src_croped_circle;
	Point center1(cvRound(targets.back().front()[0]), cvRound(targets.back().front()[1]));
	int radius1 = cvRound(targets.back().front()[2]);
	Point center2(cvRound(targets.back().back()[0]), cvRound(targets.back().back()[1]));
	int radius2 = cvRound(targets.back().back()[2]);
	Mat roi = Mat(src, Rect(center1.x - radius1, center1.y - radius1, radius1 * 2, center2.y - center1.y + 2 * radius1));
	//Mat mask = Mat::ones(image.size(), CV_8UC1);
	roi.copyTo(src_croped_circle);
	return src_croped_circle;
}

vector<Vec3f> detectBlueCircles(Mat src) {
	Mat src_red_1, src_red_2, src_blue, src_yellow, tmp;
	inRange(src, Scalar(90, 100, 70), Scalar(120, 255, 255), src_blue);
	cvtColor(src, tmp, CV_HSV2BGR);
	GaussianBlur(src_blue, src_blue, Size(), 2, 2);
	vector<Vec3f> circles;
	vector<Vec3f> tmpCircles;
	for (int i = min(src_blue.cols/2, src_blue.rows/2); i > 250; i -= 30)
	{
		tmpCircles.clear();
		HoughCircles(src_blue, tmpCircles, CV_HOUGH_GRADIENT, 1, i * 2, 100, 40, i, i+30);
		if (tmpCircles.size() != 0)
		{
			circles.insert(circles.end(), tmpCircles.begin(), tmpCircles.end());
		}
	}
	sort(circles.begin(), circles.end(), [](Vec3f a, Vec3f b) {
		return a[2] > b[2]; });
	for (int i = 0; i < 3; i++)
	{
		if (circles.size() < 3)
			break;
		Point center1(cvRound(circles[i][0]), cvRound(circles[i][1]));
		auto j = circles.begin() + i + 1;
		while (j != circles.end())
		{
			Point center2(cvRound((*j)[0]), cvRound((*j)[1]));
			if (sqrt((center1.x - center2.x)*(center1.x - center2.x) + (center1.y - center2.y)*(center1.y - center2.y)) < 50)
			{
				j = circles.erase(j);
			}
			else
				++j;
		}
	}
	sort(circles.begin(), circles.end(), [](Vec3f a, Vec3f b) {
		return a[1] < b[1]; });
	return circles;
}

vector<Vec3f> detectRedCircles(Mat src) {
	Mat src_red_1, src_red_2, src_blue, src_yellow, tmp;
	inRange(src_hsv, Scalar(0, 100, 70), Scalar(15, 255, 255), src_red_1);
	inRange(src_hsv, Scalar(160, 100, 70), Scalar(179, 255, 255), src_red_2);
	addWeighted(src_red_1, 0.5, src_red_2, 0.5, 0, src_blue);
	cvtColor(src_hsv, tmp, CV_HSV2BGR);
	GaussianBlur(src_blue, src_blue, Size(), 3, 3);
	/*Mat element = getStructuringElement(MORPH_CROSS, Size(1, 25));
	morphologyEx(src_blue, src_blue, MORPH_CLOSE, element, Point(-1, -1), 1);
	element = getStructuringElement(MORPH_CROSS, Size(25, 1));
	morphologyEx(src_blue, src_blue, MORPH_CLOSE, element, Point(-1, -1), 1);*/
	vector<Vec3f> circles;
	Rect roi(0, 0, src_blue.cols, src_blue.rows);
	HoughCircles(src_blue(roi), circles, CV_HOUGH_GRADIENT, 1, src_blue.rows / 4, 150,40, src_blue.cols / 8, src_blue.cols / 2);
	sort(circles.begin(), circles.end(), [](Vec3f a, Vec3f b) {
		return a[1] < b[1]; });
	return circles;
}

vector<Vec3f> detectYellowCircles(Mat src) {
	Mat src_red_1, src_red_2, src_blue, src_yellow, tmp;
	inRange(src, Scalar(20, 80, 60), Scalar(40, 255, 255), src_blue);
	cvtColor(src_hsv, tmp, CV_HSV2BGR);
	GaussianBlur(src_blue, src_blue, Size(), 3, 3);
	Mat element = getStructuringElement(MORPH_CROSS, Size(1, 25));
	morphologyEx(src_blue, src_blue, MORPH_CLOSE, element, Point(-1, -1), 1);
	element = getStructuringElement(MORPH_CROSS, Size(25, 1));
	morphologyEx(src_blue, src_blue, MORPH_CLOSE, element, Point(-1, -1), 1);
	vector<Vec3f> circles;
	Rect roi(src_blue.cols / 4, 0, src_blue.cols / 2, src_blue.rows);
	HoughCircles(src_blue(roi), circles, CV_HOUGH_GRADIENT, 1, src_blue.rows/4, 100, 40, src_blue.cols / 14, src_blue.cols/6);
	sort(circles.begin(), circles.end(), [](Vec3f a, Vec3f b) {
		return a[1] < b[1]; });
	return circles;
}

void callback(int, void*)
{
	detectYellowCircles(src_hsv);
}

Mat extractArrowsMask(Mat src)
{
	Mat src_croped_circles_gray;
	Mat src_croped_circles_hsv;
	Mat src_croped_circles_mod;
	Mat src_croped_circles_edges;
	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	Mat tmp;
	vector<Mat> channels;
	cvtColor(src, tmp, CV_BGR2HSV);
	split(tmp, channels);
	channels[2].copyTo(src_croped_circles_gray);
	inRange(src_croped_circles_gray, 0, 80, src_croped_circles_gray);
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(src_croped_circles_gray, src_croped_circles_gray, MORPH_OPEN, element, Point(-1, -1), 4);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(src_croped_circles_gray, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	sort(contours.begin(), contours.end(), [](vector<Point> a, vector<Point> b) {
		return contourArea(a) > contourArea(b);
	});
	for (int j = 0; j < contours.size() && j < 8; j++)
	{
		drawContours(mask, contours, j, Scalar(255, 255, 255), -1);
	}
	return mask;
}

int main()
{
	src = imread("C:\\Projects\\Recognition\\Samples\\IMAG1069.jpg");
	Mat dst;
	if (src.cols > src.rows)
	{
		transpose(src, src);
		flip(src, src, 1);
	}
	namedWindow("Original1", WINDOW_NORMAL);
	//namedWindow("HSV", WINDOW_NORMAL);
	vector<vector<Vec3f> > targets = extractTargets(src);
	for (int i = 0; i < targets.size(); i++)
		for (int j = 0; j < targets[i].size(); j++)
		{
			Point center1(cvRound(targets[i][j][0]), cvRound(targets[i][j][1]));
			int radius1 = cvRound(targets[i][j][2]);
			circle(src, center1, radius1, Scalar(0, 0, 255), 2);
		}
	imshow("Original1", src);
	/*Mat src_croped_circle = extractTargets(src);
	Mat arrows;
	src_croped_circle.copyTo(arrows, extractArrowsMask(src_croped_circle));
	namedWindow("Test1", WINDOW_NORMAL);
	imshow("Test1", arrows);*/
	//namedWindow("Original", WINDOW_KEEPRATIO);
	//imshow("Original", src);
	waitKey(0);
	return 0;
}