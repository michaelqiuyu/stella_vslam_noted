#include "image.h"

namespace stella_vslam {
namespace util {

bool is_pure_color(const cv::Mat &image) {
    double max_gray, min_gray;
    cv::Mat temp = image.reshape(1);
    cv::minMaxLoc(temp, &min_gray, &max_gray);
    return max_gray - min_gray < 15;
}

} // namespace util
} // namespace stella_vslam
