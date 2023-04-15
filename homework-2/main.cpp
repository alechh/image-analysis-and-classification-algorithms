#include <opencv2/opencv.hpp>

/*
 * A function that reads a brightness histogram from the input image.
 * It returns an image with a brightness histogram.
 */
cv::Mat plotHistogram(const cv::Mat &image)
{
    // Создание массива для хранения гистограммы яркости
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    bool uniform = true, accumulate = false;
    cv::Mat hist;

    // Вычисление гистограммы яркости
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    // Нормализация гистограммы
    cv::normalize(hist, hist, 0, image.rows, cv::NORM_MINMAX, -1, cv::Mat());

    // Создание изображения для отображения гистограммы
    cv::Mat histImage(image.rows, histSize, CV_8UC1, cv::Scalar(255, 255, 255));

    // Нормализация гистограммы
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    // Построение гистограммы
    for (int i = 1; i < histSize; i++)
    {
        cv::line(histImage, cv::Point(i, histImage.rows),
                 cv::Point(i, histImage.rows - cvRound(hist.at<float>(i))),
                 cv::Scalar(0, 0, 0), 1, 8, 0);
    }

    // Стягивание изображения до квадрата для удобства отображения
    cv::resize(histImage, histImage, cv::Size(512, 512));

    return histImage;
}

/*
 * This function takes an image in cv::Mat format as input and performs global
 * equalization of the histogram using the equalizeHist() function from the OpenCV library.
 */
cv::Mat equalize(const cv::Mat &image)
{
    cv::Mat equalized;
    cv::equalizeHist(image, equalized);
    return equalized;
}

/*
 * This function inputs an image in cv::Mat format and an integer parameter size,
 * which sets the block size for Adaptive Local Histogram Equalization (CLAHE).
 * The function performs adaptive local histogram equalization using
 * the CLAHE (Contrast Limited Adaptive Histogram Equalization) method with
 * a fixed histogram clipping threshold equal to 4 and a specified block size.
 */
cv::Mat localEqualize(const cv::Mat &image, const int size)
{
    cv::Mat equalized;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    clahe->setTilesGridSize(cv::Size(size, size));
    clahe->apply(image, equalized);
    return equalized;
}

int main()
{
    // Загрузка изображения
    cv::Mat img = imread("../images/image2.JPG", cv::IMREAD_GRAYSCALE);

    // Вывод изображений
    imshow("Original image", img);

    // Построение гистограммы яркости
    cv::Mat hist = plotHistogram(img);
    imshow("Histogram", hist);

    // Эквализация
    cv::Mat equalized = equalize(img);
    imshow("Equalized image", equalized);

    cv::Mat equalizedHist = plotHistogram(equalized);
    imshow("Equalized histogram", equalizedHist);

    // Локальная эквализация
    cv::Mat localEqualized8 = localEqualize(img, 7);
    cv::Mat localEqualized30 = localEqualize(img, 30);
    imshow("Local equalized8 image", localEqualized8);
    imshow("Local equalized30 image", localEqualized30);

    cv::Mat localEqualizedHist8 = plotHistogram(localEqualized8);
    cv::Mat localEqualizedHist30 = plotHistogram(localEqualized30);

    imshow("Local equalized8 histogram", localEqualizedHist8);
    imshow("Local equalized30 histogram", localEqualizedHist30);

    cv::waitKey(0);

    imwrite("../results/original.jpg", img);
    imwrite("../results/hist.jpg", hist);
    imwrite("../results/equalized.jpg", equalized);
    imwrite("../results/equalized_hist.jpg", equalizedHist);
    imwrite("../results/local_equalized8.jpg", localEqualized8);
    imwrite("../results/local_equalized30.jpg", localEqualized30);
    imwrite("../results/local_equalized8_hist.jpg", localEqualizedHist8);
    imwrite("../results/local_equalized30_hist.jpg", localEqualizedHist30);

    return 0;
}
