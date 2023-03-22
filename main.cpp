#include "opencv2/opencv.hpp"

/*
 * This function applies the Laplace operator to sharpen the image.
 *
 * The Laplace operator is used to detect sharp changes in brightness in the image.
 * It calculates the second derivative of the image in each direction
 * and adds them together to get a value which is a measure of the change
 * in brightness at that point in the image.
 */
cv::Mat sharpenWithLaplaceOperator(const cv::Mat& image, double c)
{
    cv::Mat result;
    cv::Laplacian(image, result, CV_16S);
    cv::convertScaleAbs(result, result);
    result = image - c * result;
    result.convertTo(result, CV_8U);
    return result;
}

/*
 * This function applies a Gaussian blur to the image and then 
 * subtracts the blurred image from the original image.
 *
 * The result is a high-pass filtered image.
 * The high-pass filtered image is then added to the original image to create the sharpened image.
 *
 * The c parameter is a constant that controls the amount of sharpening.
 */
cv::Mat sharpenWithBlurredMask(const cv::Mat& inputImage, double c)
{
    cv::Mat blurredImage, highPassImage, sharpenedImage;

    // Apply Gaussian blur to the input image
    const int kernelSize = 5;
    const double sigma = 3;
    GaussianBlur(inputImage, blurredImage,
                 cv::Size(kernelSize, kernelSize),
                 sigma);

    // Subtract the blurred image from the input image
    subtract(inputImage, blurredImage, highPassImage);

    highPassImage = c * highPassImage;

    // Add the high-pass filtered image to the original image to create the sharpened image
    add(inputImage, highPassImage, sharpenedImage);

    return sharpenedImage;
}

/*
 * This function applies the Sobel operator to the image to calculate the gradient.
 *
 * The Sobel operator is used to calculate the gradient of the image.
 * The gradient is a measure of the change in brightness in the image.
 * The gradient is calculated by convolving the image with a kernel.
 */
cv::Mat gradientSobel(const cv::Mat& image, int kernelSize = 3)
{
    cv::Mat result;
    cv::Sobel(image, result, CV_16S, 1, 1, kernelSize);
    cv::convertScaleAbs(result, result);
    return result;
}


void saveLaplacian(const cv::Mat& image)
{
    cv::Mat laplacian1 = sharpenWithLaplaceOperator(image, 1);
    cv::imwrite("../results/image_laplacian1.jpg", laplacian1);

    cv::Mat laplacian2 = sharpenWithLaplaceOperator(image, 1.5);
    cv::imwrite("../results/image_laplacian1_5.jpg", laplacian2);

    cv::Mat laplacian3 = sharpenWithLaplaceOperator(image, 1.7);
    cv::imwrite("../results/image_laplacian1_7.jpg", laplacian3);

    cv::Mat laplacian4 = sharpenWithLaplaceOperator(image, 2);
    cv::imwrite("../results/image_laplacian2.jpg", laplacian4);
}

void saveSharpen(const cv::Mat& image)
{
    cv::Mat sharpen1 = sharpenWithBlurredMask(image, 1);
    cv::imwrite("../results/image3_sharpen1.jpg", sharpen1);

    cv::Mat sharpen2 = sharpenWithBlurredMask(image, 1.5);
    cv::imwrite("../results/image3_sharpen1_5.jpg", sharpen2);

    cv::Mat sharpen3 = sharpenWithBlurredMask(image, 1.7);
    cv::imwrite("../results/image3_sharpen1_7.jpg", sharpen3);

    cv::Mat sharpen4 = sharpenWithBlurredMask(image, 2);
    cv::imwrite("../results/image3_sharpen2.jpg", sharpen4);
}

void saveGradient(const cv::Mat& image, const std::string& imageName)
{
    cv::Mat gradient1 = gradientSobel(image, 3);
    cv::imwrite("../results/" + imageName + "_gradient3.jpg", gradient1);

    cv::Mat gradient2 = gradientSobel(image, 5);
    cv::imwrite("../results/" + imageName + "_gradient5.jpg", gradient2);
}

void saveCombine(const cv::Mat& src)
{
    cv::Mat laplacian;
    cv::Laplacian(src, laplacian, CV_16S);
    cv::convertScaleAbs(laplacian, laplacian);

    cv::Mat result = src + laplacian;

    cv::imshow("result", result);

    cv::Mat gradient2 = gradientSobel(src, 5);

    cv::imshow("gradient2", gradient2);

    // gauss blur to gradient2
    const int kernelSize = 5;
    const double sigma = 3;
    cv::Mat gradient2_gauss;
    GaussianBlur(gradient2, gradient2_gauss,
                 cv::Size(kernelSize, kernelSize),
                 sigma);

    cv::imshow("gradient2_gauss", gradient2_gauss);

    // result * gradient2_gauss
    cv::Mat result2;
    cv::multiply(result, gradient2_gauss, result2);

    cv::imshow("result2", result2);

    // src + result2
    cv::Mat result3 = src + result2;

    cv::imshow("result3", result3);
    cv::waitKey(0);

    //cv::imwrite("../results/image3_combine_laplacian.jpg", laplacian);
    //cv::imwrite("../results/image3_combine_src_lapl.jpg", result);
    //cv::imwrite("../results/image3_combine_gradient.jpg", gradient2);
}

int main()
{
    cv::Mat image = cv::imread("../images/image3.jpeg");
    cv::imshow("Original", image);

    //saveLaplacian(image);

    //saveSharpen(image);

    //saveGradient(image, "image2");

    saveCombine(image);

    cv::waitKey(0);
    return 0;
}
