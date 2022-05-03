#include <iostream>
#include <chrono>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char *argv[])
{
    cv::Mat image;
    if (argc == 1){
        cerr << "至少要有一个参数" << endl;
        return 0;
    }
    image = cv::imread(argv[1]);
    if (image.data == nullptr) {
        cerr << "文件" << argv[1] << "不存在." << endl;
        return 0;
    }
    cout << "图像宽为" << image.cols << ", 高为" << image.rows << "，通道数为" << image.channels() << endl;
	//cv::imshow("image", image);
    //cv::waitKey(0);
    cout << image.type() << endl;
    cout << CV_8UC1 << endl;
    cout << CV_8UC3 << endl;
    cv::Mat image_another = image;
    image_another(cv::Rect(0, 0, 100, 100)).setTo(0);
    cv::imshow("image", image);
    cv::waitKey(0);
    return 0;
}
