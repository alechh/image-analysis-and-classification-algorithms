#include <opencv2/opencv.hpp>

/*
* This function centers and logarithms the frequency spectrum of the image in the frequency domain.
*/
cv::Mat getCenteredFourierSpectrum(cv::Mat inputImage)
{
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(inputImage.rows);
    int n = cv::getOptimalDFTSize(inputImage.cols);
    copyMakeBorder(inputImage, padded, 0, m - inputImage.rows, 0, n - inputImage.cols,
                   cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImage;
    merge(planes, 2, complexImage);
    dft(complexImage, complexImage);
    split(complexImage, planes);
    magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magImage = planes[0];
    magImage += cv::Scalar::all(1);
    log(magImage, magImage);
    magImage = magImage(cv::Rect(0, 0, magImage.cols & -2, magImage.rows & -2));
    int cx = magImage.cols / 2;
    int cy = magImage.rows / 2;
    cv::Mat q0(magImage, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magImage, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magImage, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magImage, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(magImage, magImage, 0, 255, cv::NORM_MINMAX);
    cv::Mat outputImage;
    magImage.convertTo(outputImage, CV_8U);
    return outputImage;
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

/*
 * This is a function that takes an input image and returns its Fourier spectrum.
 */
cv::Mat getFourierSpectrum(cv::Mat inputImage)
{
    // Convert the input image to grayscale
    cv::Mat grayImage;
    cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    // Calculate the size of the padded image for FFT
    int rows = cv::getOptimalDFTSize(grayImage.rows);
    int cols = cv::getOptimalDFTSize(grayImage.cols);

    // Create a new image with padded size
    cv::Mat paddedImage;
    cv::copyMakeBorder(grayImage, paddedImage, 0, rows - grayImage.rows, 0,
                       cols - grayImage.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Create a complex image for storing the FFT output
    cv::Mat complexImage;
    cv::dft(cv::Mat_<float>(paddedImage), complexImage, cv::DFT_COMPLEX_OUTPUT);

    // Shift the image so that the low frequencies are in the center
    cv::Mat shiftedImage;
    fftShift(complexImage, shiftedImage);

    // Calculate the magnitude spectrum (x, y, magnitude)
    cv::Mat magnitudeImages[2];
    cv::Mat magnitudeImage;
    cv::split(shiftedImage, magnitudeImages);
    cv::magnitude(magnitudeImages[0], magnitudeImages[1], magnitudeImage);

    // Convert the magnitude spectrum to logarithmic scale
    cv::Mat logMagnitudeImage;
    cv::log(1 + magnitudeImage, logMagnitudeImage);

    // Normalize the logarithmic magnitude spectrum
    cv::normalize(logMagnitudeImage, logMagnitudeImage, 0, 1, cv::NORM_MINMAX);

    return logMagnitudeImage;
}

/*
 * Notch filter, which nulls the F(0,0) Fourier-transform term.
 */
cv::Mat applyNotchFilter(const cv::Mat& inputImage)
{
    cv::Mat outputImage;
    cv::Mat complexImage;
    cv::Mat filter;

    // Convert the image into a complex space
    dft(inputImage, complexImage, cv::DFT_COMPLEX_OUTPUT);

    // Creating a notch filter
    filter = cv::Mat::ones(complexImage.size(), CV_32FC2);
    filter.at<cv::Vec2f>(0, 0) = cv::Vec2f(0, 0);

    // Applying the filter
    mulSpectrums(complexImage, filter, complexImage, 0);

    // Inverse Fourier Transform
    idft(complexImage, outputImage, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    return outputImage;
}

int main()
{
    const std::string imageName = "dog";
    cv::Mat src = cv::imread("../images/" + imageName + ".jpeg", cv::IMREAD_GRAYSCALE);

    cv::imshow("Source image", src);
    cv::imwrite("../results/" + imageName + "_src.jpeg", src);

    cv::Mat src32;
    src.convertTo(src32, CV_32FC1);

    const cv::Mat mag = applyNotchFilter(src32);
    cv::imshow("Result", mag);
    cv::waitKey(0);

    // Prepare to save
    cv::Mat mag8;
    mag.convertTo(mag8, CV_8UC1);

    cv::imwrite("../results/" + imageName + "_notch_filter.jpeg", mag8);

    return 0;
}
