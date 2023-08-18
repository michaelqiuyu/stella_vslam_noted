#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/io/map_database_io_base.h"
#include "stella_vslam/data/map_database.h"
#include "stella_vslam/test_macro.h"

#include <spdlog/spdlog.h>
#include <fstream>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>

namespace stella_vslam {
namespace io {


// note: by xiongchao
void map_database_io_base::save_pose_txt(const std::string& path, data::map_database* map_db) {
    std::ofstream ofs(path);
    if (ofs.is_open()) {
        ofs << std::fixed << std::setprecision(6);
        ofs << "# frame_id x y z qx qy qz qw" << std::endl;

        const auto keyfrms = map_db->get_all_keyframes();
        for (const auto& keyfrm : keyfrms) {
            auto Twc = keyfrm->get_pose_wc();
            Eigen::Quaterniond q(Twc.block<3, 3>(0, 0));

            int frame_id = round(keyfrm->timestamp_ * keyfrm->camera_->fps_);
            ofs << frame_id << " "
                << Twc(0, 3) << " " << Twc(1, 3) << " " << Twc(2, 3) << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                << std::endl;
        }
    } else {
        spdlog::critical("cannot create a file at {}", path);
    }

#ifdef SAVE_KEYFRAME_IMAGE
    std::string kf_images = "/home/xiongchao/workspace/leador/project/vslam/dataset/leador/reloc_test/outdoor/front_kf_images/";

    const auto keyfrms = map_db->get_all_keyframes();
    for (const auto& keyfrm : keyfrms) {
        int frame_count = int(round(keyfrm->timestamp_ * keyfrm->camera_->fps_));
        std::string image_name = kf_images + "/" + std::to_string(frame_count) + ".jpg";
        cv::imwrite(image_name, keyfrm->get_img());
    }
#endif
}




}
}
