#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/io/map_database_io_base.h"
#include "stella_vslam/data/map_database.h"

#include <spdlog/spdlog.h>
#include <fstream>

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
}




}
}
