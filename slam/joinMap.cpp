#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace std;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
//Vector6d的定义
typedef Eigen::Matrix<double, 6, 1> Vector6d;

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char *argv[])
{
    vector<cv::Mat> colorImgs, depthImgs;
    //下面是一个SE3d的向量
    TrajectoryType poses;
    ifstream fin("./pose.txt");
    if(!fin){
        cerr << "请在有pose.txt的目录下运行此程序" << endl;
        return 1;
    }
    //读数据
    for(int i = 0; i < 5; i ++){
        // boost format类来进行格式化字符串
        boost::format fmt("./%s/%d.%s");
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1 ) % "pgm").str(), -1)); //使用-1读取原始图像？ 是什么意思？
        //opencv 是可以读出pgm图片的

        double data[7] = {0};
        //从pose.txt中读取7个数字
        for (auto &d : data)
            fin >> d;
        //SE3d的构造方法是 用一个Quaterniond和一个Vector3d来进行构造
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(pose);
    }

    // 计算点云并拼接
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0; // 这个depthScale 有什么用呢
    //因为对于16位的整型 最小值是1，需要有一个比例来获得1以下的深度
    // 下面这个就等价于 vector<Vector6d> 语法上要加上后面的
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    // 这个reserve函数使得vector的容量至少能够容纳n个元素
    pointcloud.reserve(1000000);
    for(int i = 0;i < 5; i ++){
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Sophus::SE3d T = poses[i];
        for (int v= 0; v < color.rows; v ++){
            for(int u = 0; u < color.cols; u ++){
                // 这个depth.ptr是个什么操作
                unsigned int d = depth.ptr<unsigned short>(v)[u];
                if (d == 0) continue;
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;
                Vector6d p;
                p.head<3>() = pointWorld;
                p[5] = color.data[v * color.step + u * color.channels()];
                p[4] = color.data[v * color.step + u * color.channels() + 1];
                p[3] = color.data[v * color.step + u * color.channels() + 2];
                pointcloud.push_back(p);
            }
        }
    }
    cout << "点云共有" << pointcloud.size() << "个点。" << endl;
    showPointCloud(pointcloud);
    return 0;
}

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3] / 255.0, p[4] / 255.0, p[5] /255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}
