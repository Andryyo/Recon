#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat original_image, src_gray, croped_src, src_hsv;
Rect targetROI;

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
	cvtColor(image, src_hsv, CV_BGR2HSV);
	vector<Mat> HSV;
	split(src_hsv, HSV);
	auto clahe = createCLAHE(3.0, Size(8, 8));
	clahe->apply(HSV[2], HSV[2]);
	merge(HSV, src_hsv);
	cvtColor(src_hsv, image, CV_HSV2BGR);
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
	int border = 40;
	targetROI = Rect(center1.x - radius1 - border, center1.y - radius1 - border, radius1 * 2 + border * 2, center2.y - center1.y + 2 * radius1 + border * 2);
	Mat roi = Mat(src, targetROI);
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

vector<Vec3f> detectYellowCircles(Mat src)	 {
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
	//equalizeHist(channels[2], channels[2]);
	channels[2].copyTo(src_croped_circles_gray);
	inRange(src_croped_circles_gray, 0, 90, src_croped_circles_gray);
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(src_croped_circles_gray, src_croped_circles_gray, MORPH_OPEN, element, Point(-1, -1), 4);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(src_croped_circles_gray, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	sort(contours.begin(), contours.end(), [](vector<Point> a, vector<Point> b) {
		return contourArea(a) > contourArea(b);
	});
	for (int j = 0; j < contours.size() && contourArea(contours[j]) > 2000; j++)
	{
		drawContours(mask, contours, j, Scalar(255, 255, 255), -1);
	}
	return mask;
}

vector<Point> extractPoints(Mat src)
{
	Mat canny;
	vector<Point> result;
	//Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	//morphologyEx(src, canny, MORPH_OPEN, element, Point(-1, -1), 4);
	GaussianBlur(src, canny, Size(), 2, 2);
	Canny(canny, canny, 100, 200);
	vector<Vec2f> lines;
	HoughLines(canny, lines, 1, CV_PI / 180, 100);
	sort(lines.begin(), lines.end(), [](Vec2f a, Vec2f b) {
		return a[1] > b[1];
	});
	vector<vector<Point>> points(lines.size());
	for (int i = 0; i < lines.size(); i++)
	{
		double r1 = lines[i][0];
		double s1 = sin(lines[i][1]);
		double c1 = cos(lines[i][1]);
		for (int j = 0; j < canny.cols; ++j)
		{
			int y1 = (r1 - j * c1) / s1;
			if (y1 < 0 || y1 >= canny.rows)
				continue;
			if (canny.at<unsigned char>(y1, j) == 255)
			{
				points[i].push_back(Point(j, y1));
				circle(original_image(targetROI), Point(j, y1), 3, Scalar(0, 255, 0), 2);
			}
		}
	}
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 4000 * (-b));
		pt1.y = cvRound(y0 + 4000 * (a));
		pt2.x = cvRound(x0 - 4000 * (-b));
		pt2.y = cvRound(y0 - 4000 * (a));
		line(original_image(targetROI), pt1, pt2, Scalar(0, 0, 255), 1, CV_AA);
	}
	for (int i = 0; i < lines.size() - 1; i++)
	{
		for (int j = i + 1; j < lines.size(); j++)
		{
			double r1 = lines[i][0];
			double r2 = lines[j][0];
			double s1 = lines[i][1];
			double s2 = lines[j][1];
			double c1 = lines[i][1];
			double c2 = lines[j][1];
			int x0 = (r2 * s1 - r1 * s2) / (c2 * s1 - c1 * s2);
			int y0 = (r1 - x0 * c1) / s1;
			//circle(original_image(targetROI), Point(x, y), 5, Scalar(0, 255, 0), 3);
			//circle(original_image(targetROI), points[i], 3, Scalar(0, 255, 0), 2);
			//circle(original_image(targetROI), lines[j], 3, Scalar(0, 255, 0), 2);
		} 
	}
	namedWindow("Points", WINDOW_NORMAL);
	imshow("Points", original_image);
	return result;
}

int main()
{
	original_image = imread("C:\\Projects\\Recognition\\Samples\\IMAG1068.jpg");
	Mat dst;
	if (original_image.cols > original_image.rows)
	{
		transpose(original_image, original_image);
		flip(original_image, original_image, 1);
	}
	namedWindow("Original1", WINDOW_NORMAL);
	//namedWindow("HSV", WINDOW_NORMAL);
	vector<vector<Vec3f> > targets = extractTargets(original_image);
	imshow("Original1", original_image);
	Mat src_croped_circle = cutTargets(original_image, targets);
	Mat mask = extractArrowsMask(src_croped_circle);
	vector<Point> marks = extractPoints(mask);
	for (int i = 0; i < targets.size(); i++)
		for (int j = 0; j < targets[i].size(); j++)
		{
			Point center1(cvRound(targets[i][j][0]), cvRound(targets[i][j][1]));
			int radius1 = cvRound(targets[i][j][2]);
			circle(original_image, center1, radius1, Scalar(0, 0, 255), 2);
		}
	imshow("Points", original_image);
	//src_croped_circle.copyTo(arrows, extractArrowsMask(src_croped_circle));
	//namedWindow("Test1", WINDOW_NORMAL);
	//imshow("Test1", arrows);
	//namedWindow("Original", WINDOW_KEEPRATIO);
	//imshow("Original", src);
	waitKey(0);
	return 0;
}