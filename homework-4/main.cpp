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


/*
 * Combining image sharpening methods: Laplace operator, unsharp masking and Sobel operator.
 */
cv::Mat combineMethods(const cv::Mat& image)
{
    using namespace cv;

    Mat sharpen_kernel = (Mat_<float>(3,3) <<
                                           -1, -1, -1,
            -1,  9, -1,
            -1, -1, -1);

    Mat sharpened;
    filter2D(image, sharpened, image.depth(), sharpen_kernel);

    Mat grad;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    Sobel(image, grad_x, image.depth(), 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(image, grad_y, image.depth(), 0, 1, 3, 1, 0, BORDER_DEFAULT);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    Mat laplacian;
    Laplacian(image, laplacian, image.depth(), 3, 1, 0, BORDER_DEFAULT);

    Mat sharpened_image;

    addWeighted(image, 1.0, laplacian, -0.5, 0, sharpened_image);

    addWeighted(sharpened_image, 1.0, grad, 0.7, 0, sharpened_image);

    return sharpened_image;
}

int main()
{
    cv::Mat image = cv::imread("../images/flowers.jpg");
    cv::imshow("Original", image);

    saveLaplacian(image);

    saveSharpen(image);

    saveGradient(image, "image2");

    cv::Mat dst = combineMethods(image);
    cv::imwrite("../results/flowers_sharpened.jpg", dst);

    return 0;
}
