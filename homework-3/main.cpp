#include <opencv2/opencv.hpp>

/*
 * Image filtering function using a simple average filter.
 *
 * The idea of the filter is to go through each pixel of the image,
 * and calculate the average of the colors from the pixels in the vicinity
 * of that pixel using a kernel of a given size.
 */
void filter(const cv::Mat &src, cv::Mat &dst, int kernel_size)
{
    int kernel_half = kernel_size / 2;
    int kernel_area = kernel_size * kernel_size;

    for (int y = kernel_half; y < src.rows - kernel_half; y++)
    {
        for (int x = kernel_half; x < src.cols - kernel_half; x++)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int i = -kernel_half; i <= kernel_half; i++)
            {
                for (int j = -kernel_half; j <= kernel_half; j++)
                {
                    cv::Vec3b color = src.at<cv::Vec3b>(y + i, x + j);
                    sum_b += color[0];
                    sum_g += color[1];
                    sum_r += color[2];
                }
            }

            cv::Vec3b color;
            color[0] = sum_b / kernel_area;
            color[1] = sum_g / kernel_area;
            color[2] = sum_r / kernel_area;

            dst.at<cv::Vec3b>(y, x) = color;
        }
    }
}

/*
 * Image filtering function using a weighted average filter.
 *
 * The idea of the filter is to pass through each pixel of the image,
 * and calculate a weighted average of the colors from the pixels in the vicinity
 * of that pixel using a kernel of a given size.
 *
 * The weight of each pixel in the vicinity depends on the distance to the center of the kernel.
 */
void filterWeightedAverage(const cv::Mat &src, cv::Mat &dst, int kernel_size)
{
    int kernel_half = kernel_size / 2;
    int kernel_area = kernel_size * kernel_size;

    for (int y = kernel_half; y < src.rows - kernel_half; y++)
    {
        for (int x = kernel_half; x < src.cols - kernel_half; x++)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int i = -kernel_half; i <= kernel_half; i++)
            {
                for (int j = -kernel_half; j <= kernel_half; j++)
                {
                    cv::Vec3b color = src.at<cv::Vec3b>(y + i, x + j);
                    sum_b += color[0] * (kernel_half - abs(i)) * (kernel_half - abs(j));
                    sum_g += color[1] * (kernel_half - abs(i)) * (kernel_half - abs(j));
                    sum_r += color[2] * (kernel_half - abs(i)) * (kernel_half - abs(j));
                }
            }

            cv::Vec3b color;
            color[0] = sum_b / kernel_area;
            color[1] = sum_g / kernel_area;
            color[2] = sum_r / kernel_area;

            dst.at<cv::Vec3b>(y, x) = color;
        }
    }
}

/*
* This function applies median filtering to an input image.
* Median filtering is a type of nonlinear filtering that replaces each pixel
* value with the median value of the neighboring pixels within a specified kernel size.
*/
void medianFilter(const cv::Mat &src, cv::Mat &dst, int kernel_size)
{
    int kernel_half = kernel_size / 2;
    int kernel_area = kernel_size * kernel_size;

    for (int y = kernel_half; y < src.rows - kernel_half; y++)
    {
        for (int x = kernel_half; x < src.cols - kernel_half; x++)
        {
            std::vector<int> b(kernel_area);
            std::vector<int> g(kernel_area);
            std::vector<int> r(kernel_area);

            int k = 0;
            for (int i = -kernel_half; i <= kernel_half; i++)
            {
                for (int j = -kernel_half; j <= kernel_half; j++)
                {
                    cv::Vec3b color = src.at<cv::Vec3b>(y + i, x + j);
                    b[k] = color[0];
                    g[k] = color[1];
                    r[k] = color[2];
                    k++;
                }
            }

            std::sort(b.begin(), b.end());
            std::sort(g.begin(), g.end());
            std::sort(r.begin(), r.end());

            cv::Vec3b color;
            color[0] = b[kernel_area / 2];
            color[1] = g[kernel_area / 2];
            color[2] = r[kernel_area / 2];

            dst.at<cv::Vec3b>(y, x) = color;
        }
    }
}

int main()
{
    std::string image = "image4.jpeg";
    cv::Mat src = cv::imread("../images/" + image);
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    cv::Mat dst2 = cv::Mat::zeros(src.size(), src.type());
    cv::Mat dst3 = cv::Mat::zeros(src.size(), src.type());

    filter(src, dst, 5);
    filterWeightedAverage(src, dst2, 5);
    medianFilter(src, dst3, 7);

    cv::imshow("src", src);
    cv::imshow("Awerage filter", dst);
    cv::imshow("Weighted average filter", dst2);
    cv::imshow("Median filter", dst3);
    cv::waitKey(0);

    cv::imwrite("../results/" + image + "_awerage_filtered.jpg", dst);
    cv::imwrite("../results/" + image + "_weighted_average_filtered.jpg", dst2);
    cv::imwrite("../results/" + image + "_median_filtered.jpg", dst3);

    return 0;
}
