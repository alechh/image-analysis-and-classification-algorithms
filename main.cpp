#include <iostream>
#include "opencv2/opencv.hpp"

/*
 * Negative of an image
 */
cv::Mat negative(const cv::Mat &src)
{
    cv::Mat dst = src.clone();

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            dst.at<cv::Vec3b>(i, j)[0] = 255 - dst.at<cv::Vec3b>(i, j)[0];
            dst.at<cv::Vec3b>(i, j)[1] = 255 - dst.at<cv::Vec3b>(i, j)[1];
            dst.at<cv::Vec3b>(i, j)[2] = 255 - dst.at<cv::Vec3b>(i, j)[2];
        }
    }

    return dst;
}

/*
 * Logarithmic transformation
 */
cv::Mat logarithmic(const cv::Mat &src)
{
    cv::Mat dst = src.clone();

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            dst.at<cv::Vec3b>(i, j)[0] = 255 * log(1 + dst.at<cv::Vec3b>(i, j)[0]) / log(256);
            dst.at<cv::Vec3b>(i, j)[1] = 255 * log(1 + dst.at<cv::Vec3b>(i, j)[1]) / log(256);
            dst.at<cv::Vec3b>(i, j)[2] = 255 * log(1 + dst.at<cv::Vec3b>(i, j)[2]) / log(256);
        }
    }

    return dst;
}

/*
 * Stepwise transformation
 */
cv::Mat stepwise(const cv::Mat &src, const double c, const double gamma)
{
    cv::Mat dst = src.clone();

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            dst.at<cv::Vec3b>(i, j)[0] = c * pow(dst.at<cv::Vec3b>(i, j)[0], gamma);
            dst.at<cv::Vec3b>(i, j)[1] = c * pow(dst.at<cv::Vec3b>(i, j)[1], gamma);
            dst.at<cv::Vec3b>(i, j)[2] = c * pow(dst.at<cv::Vec3b>(i, j)[2], gamma);
        }
    }

    return dst;
}

void contrastEnhancementInPixelChannel(uchar &pixel)
{
    if (pixel < 64)
    {
        pixel = 4 * pixel;
    }
    else if (pixel < 128)
    {
        pixel = 255 / 3;
    }
    else if (pixel)
    {
        pixel = 255 / 3 + 4 * (pixel - 128);
    }
    else
    {
        pixel = 255;
    }
}

/*
 * Contrast enhancement with a piecewise linear function
*/
cv::Mat contrastEnhancement(const cv::Mat &src)
{
    cv::Mat dst = src.clone();

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            contrastEnhancementInPixelChannel(dst.at<cv::Vec3b>(i, j)[0]);
            contrastEnhancementInPixelChannel(dst.at<cv::Vec3b>(i, j)[1]);
            contrastEnhancementInPixelChannel(dst.at<cv::Vec3b>(i, j)[2]);
        }
    }

    return dst;
}

/*
 * Linear transformation to increase image brightness
 */
cv::Mat brightnessIncrease(const cv::Mat &src, const int value)
{
    cv::Mat dst = src.clone();

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            dst.at<cv::Vec3b>(i, j)[0] = dst.at<cv::Vec3b>(i, j)[0] + value > 255 ? 255 : dst.at<cv::Vec3b>(i, j)[0] + value;
            dst.at<cv::Vec3b>(i, j)[1] = dst.at<cv::Vec3b>(i, j)[1] + value > 255 ? 255 : dst.at<cv::Vec3b>(i, j)[1] + value;
            dst.at<cv::Vec3b>(i, j)[2] = dst.at<cv::Vec3b>(i, j)[2] + value > 255 ? 255 : dst.at<cv::Vec3b>(i, j)[2] + value;
        }
    }

    return dst;
}


int main() {
    cv::Mat src, dst;

    std::cout << "Choose the transformation you want to apply:" << std::endl;
    std::cout << "1. Negative" << std::endl;
    std::cout << "2. Logarithmic" << std::endl;
    std::cout << "3. Stepwise" << std::endl;
    std::cout << "4. Contrast enhancement" << std::endl;
    std::cout << "5. Brightness increase" << std::endl;
    std::cout << "Option: ";

    int menu;
    std::cin >> menu;

    if (menu == 1)
    {
        src = cv::imread("../images/image1.JPG");
        dst = negative(src);
    }
    else if (menu == 2)
    {
        src = cv::imread("../images/image1.JPG");
        dst = logarithmic(src);
    }
    else if (menu == 3)
    {
        src = cv::imread("../images/image1.JPG");
        dst = stepwise(src, 1, 0.5);
    }
    else if (menu == 4)
    {
        src = cv::imread("../images/image1.JPG");
        dst = contrastEnhancement(src);
    }
    else if (menu == 5)
    {
        src = cv::imread("../images/image1.JPG");
        dst = brightnessIncrease(src, 60);
    }
    else
    {
        std::cout << "Invalid option" << std::endl;
        return -1;
    }

    cv::imshow("Source", src);
    cv::imshow("Result", dst);
    cv::waitKey(0);

    return 0;
}
