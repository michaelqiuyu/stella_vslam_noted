#ifndef STELLA_VSLAM_IMAGE_H
#define STELLA_VSLAM_IMAGE_H

#include <opencv2/core.hpp>

namespace stella_vslam {
namespace util {

bool is_pure_color(const cv::Mat &image);

} // namespace util
} // namespace stella_vslam

#endif //STELLA_VSLAM_IMAGE_H
