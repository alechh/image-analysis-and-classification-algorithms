#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

using namespace cv;

cv::Mat ideal_LPF(const cv::Mat& img, int D)
{
    // Преобразование в частотную область
    cv::Mat imgFreq = img.clone();
    imgFreq.convertTo(imgFreq, CV_32F);
    dft(imgFreq, imgFreq, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat tmp = imgFreq.clone();

    int M = tmp.rows;
    int N = tmp.cols;
    int cx = M / 2;
    int cy = N / 2;

    // Проход по всему изображению
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // Расстояние до центра частотной области
            double Duv = sqrt(pow(i - cx, 2) + pow(j - cy, 2));

            // Идеальный низкочастотный фильтр
            if (Duv > D)
            {
                tmp.at<cv::Vec3b>(i, j)[0] = 0;
                tmp.at<cv::Vec3b>(i, j)[1] = 0;
                tmp.at<cv::Vec3b>(i, j)[2] = 0;
            }
        }
    }

    // Обратное преобразование Фурье
    cv::Mat res;
    idft(tmp, res, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    //res.convertTo(res, CV_8U);

    return res;
}

Mat idealLowPassFilter(Mat img, int D)
{
    // Применяем быстрое преобразование Фурье
    Mat imgFreq;
    cv::dft(img, imgFreq, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);

    // Получаем размеры изображения
    int M = img.rows;
    int N = img.cols;

    // Создаем фильтр
    Mat H = Mat::zeros(M, N, CV_32FC2);
    Point2f center(N / 2.0f, M / 2.0f);
    float D2 = D * D;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float distance2 = pow(j - center.x, 2) + pow(i - center.y, 2);
            if (distance2 <= D2)
            {
                H.at<Vec2f>(i, j)[0] = 1.0f;
                H.at<Vec2f>(i, j)[1] = 0.0f;
            }
        }
    }

    // Применяем фильтр
    Mat imgFilteredFreq;
    cv::mulSpectrums(imgFreq, H, imgFilteredFreq, cv::DFT_COMPLEX_OUTPUT);

    // Обратное быстрое преобразование Фурье
    Mat imgFiltered;
    cv::idft(imgFilteredFreq, imgFiltered, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    // Приводим изображение к типу CV_8U
    imgFiltered.convertTo(imgFiltered, CV_8U);

    return imgFiltered;
}


void butterworthFilter(Mat& src, Mat& dst, int d, int n)
{
    Mat padded;
    int w = getOptimalDFTSize(src.cols);
    int h = getOptimalDFTSize(src.rows);
    copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI, complexI);

    float D;
    float D0 = d;
    float N = n;

    for (int i = 0; i < complexI.rows; i++)
    {
        for (int j = 0; j < complexI.cols; j++)
        {
            D = sqrt(pow((i - complexI.rows / 2), 2) + pow((j - complexI.cols / 2), 2));
            float H = 1 / (1 + pow(D / D0, 2 * N));
            complexI.at<Vec2f>(i, j)[0] *= H;
            complexI.at<Vec2f>(i, j)[1] *= H;
        }
    }

    idft(complexI, complexI);
    split(complexI, planes);
    normalize(planes[0], dst, 0, 255, NORM_MINMAX);
    dst.convertTo(dst, CV_8UC1);
}

/*
 * This function is used to shift the image such that the low
 * frequencies are in the center of the image.
 */
void fftShift(const cv::Mat& inputImg, cv::Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    cv::Mat q0(outputImg, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(outputImg, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(outputImg, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(outputImg, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

Mat applyGaussFilter(Mat& input, double cutoffFrequency)
{
    // Подсчет частотной таблицы
    Mat fourierTransform;
    cv::dft(input, fourierTransform, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
    Mat shifted;
    fftShift(fourierTransform, shifted);

    // Получение размеров изображения
    int rows = shifted.rows;
    int cols = shifted.cols;
    int midX = cols / 2;
    int midY = rows / 2;

    // Подсчет маски Гаусса
    Mat gaussMask(rows, cols, CV_32FC2);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double d = sqrt(pow((j - midX), 2) + pow((i - midY), 2));
            double g = exp(-(pow(d, 2) / (2 * pow(cutoffFrequency, 2))));
            gaussMask.at<Vec2f>(i, j)[0] = g;
            gaussMask.at<Vec2f>(i, j)[1] = g;
        }
    }

    // Применение маски Гаусса
    Mat filteredImage;
    cv::mulSpectrums(shifted, gaussMask, filteredImage, cv::DFT_ROWS);
    cv::idft(filteredImage, filteredImage, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::normalize(filteredImage, filteredImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    return filteredImage;
}

int main()
{
    const std::string imageName = "image3";
    cv::Mat img = cv::imread("../images/" + imageName + ".jpeg", cv::IMREAD_GRAYSCALE);
    cv::imshow("Original image", img);

    img.convertTo(img, CV_32FC1);

    const int D = 100;

    Mat img_filtered;
    butterworthFilter(img, img_filtered, D, 2);

    imshow("Filtered image (D = )",  img_filtered);

    cv::waitKey(0);
    return 0;
}
