#include <iostream>
#include "opencv2/opencv.hpp"

/*
 * Negative of an image.
 *
 * The function returns a negative, that is, an image in which each pixel has been replaced by its inverted color
 * (255 minus the value of each channel in the pixel).
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
 *
 * This function implements a logarithmic image brightness correction algorithm,
 * which is used to improve contrast in an image where pixel values are shifted to darker tones.
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
 *
 * This function implements a threshold image brightness correction algorithm that applies
 * a non-linear transformation to the brightness of each pixel based on the threshold value and gamma parameter.
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

/*
 * Linear transformation to increase image brightness
 *
 * This function performs a brightness increase operation on the input image by adding
 * a specified value to the intensity values of each pixel.
 * The function takes two arguments, the input image src and the value by which the brightness is increased.
 */
cv::Mat brightnessIncrease(const cv::Mat &src, const int value)
{
    cv::Mat dst = src.clone();

    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            dst.at<cv::Vec3b>(i, j)[0] = dst.at<cv::Vec3b>(i, j)[0] + value > 255 ?
                    255 : dst.at<cv::Vec3b>(i, j)[0] + value;
            dst.at<cv::Vec3b>(i, j)[1] = dst.at<cv::Vec3b>(i, j)[1] + value > 255 ?
                    255 : dst.at<cv::Vec3b>(i, j)[1] + value;
            dst.at<cv::Vec3b>(i, j)[2] = dst.at<cv::Vec3b>(i, j)[2] + value > 255 ?
                    255 : dst.at<cv::Vec3b>(i, j)[2] + value;
        }
    }

    return dst;
}

/*
 * The function takes an input image src in OpenCV Mat format and two parameters alpha and beta.
 * The parameters alpha and beta are used to determine the boundary values between the two parts
 * of the linear function, which is used to change the contrast of the image.
 */
cv::Mat piecewiseLinearTransform(const cv::Mat& src, float alpha, float beta)
{
    cv::Mat dst = src.clone();

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            cv::Vec3b pixel_value = src.at<cv::Vec3b>(i, j);
            for (int k = 0; k < 3; k++)
            {
                if (pixel_value[k] < 127)
                {
                    pixel_value[k] = alpha * pixel_value[k];
                } else
                {
                    pixel_value[k] = beta * pixel_value[k] + (1 - beta) * 255;
                }
            }
            dst.at<cv::Vec3b>(i, j) = pixel_value;
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
    std::cout << "4. Brightness increase" << std::endl;
    std::cout << "5. Piecewise linear transformation" << std::endl;
    std::cout << "Option: ";

    int menu;
    std::cin >> menu;

    switch (menu)
    {
        case 1:
            src = cv::imread("../images/image1.JPG");
            dst = negative(src);
            break;
        case 2:
            src = cv::imread("../images/image1.JPG");
            dst = logarithmic(src);
            break;
        case 3:
            src = cv::imread("../images/image1.JPG");
            dst = stepwise(src, 1, 0.8);
            break;
        case 4:
            src = cv::imread("../images/image1.JPG");
            dst = brightnessIncrease(src, 60);
            break;
        case 5:
            src = cv::imread("../images/image1.JPG");
            dst = piecewiseLinearTransform(src, 0.5, 0.5);
            break;
        default:
            std::cout << "Invalid option" << std::endl;
            return 0;
    }

    cv::imshow("Source", src);
    cv::imshow("Result", dst);
    cv::waitKey(0);

    cv::imwrite("../result.jpg", dst);

    return 0;
}
