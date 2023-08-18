#include "stella_vslam/match/fuse.h"
#include "stella_vslam/camera/base.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"

#include <vector>

namespace stella_vslam {
namespace match {

// 将当前关键帧的地图点往其他关键帧投影，如果其他关键帧这个特征点有地图点，那就是重复；否则就将这个点添加到这个关键帧中
template<typename T>
unsigned int fuse::detect_duplication(const std::shared_ptr<data::keyframe>& keyfrm,
                                      const Mat33_t& rot_cw,
                                      const Vec3_t& trans_cw,
                                      const T& landmarks_to_check,
                                      const float margin,
                                      std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>& duplicated_lms_in_keyfrm,
                                      std::unordered_map<unsigned int, std::shared_ptr<data::landmark>>& new_connections,
                                      bool do_reprojection_matching) const {
    const Vec3_t trans_wc = -rot_cw.transpose() * trans_cw;
    unsigned int num_fused = 0;
    std::unordered_set<unsigned int> already_matched_idx_in_keyfrm;

    duplicated_lms_in_keyfrm.clear();

    for (auto& lm : landmarks_to_check) {
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        // lm的观测中是否有这个关键帧
        if (lm->is_observed_in_keyframe(keyfrm)) {
            continue;
        }

        // 3D point coordinates with the global reference
        const Vec3_t pos_w = lm->get_pos_in_world();

        // Reproject and compute visibility
        Vec2_t reproj;
        float x_right;
        const bool in_image = keyfrm->camera_->reproject_to_image(rot_cw, trans_cw, pos_w, reproj, x_right);

        // Ignore if it is reprojected outside the image
        if (!in_image) {
            continue;
        }

        // Check if it's within ORB scale levels
        const Vec3_t cam_to_lm_vec = pos_w - trans_wc;
        const auto cam_to_lm_dist = cam_to_lm_vec.norm();
        const auto margin_far = 1.3;
        const auto margin_near = 1.0 / margin_far;
        const auto max_cam_to_lm_dist = margin_far * lm->get_max_valid_distance();
        const auto min_cam_to_lm_dist = margin_near * lm->get_min_valid_distance();

        // 在这个相机下距离相机的距离是否在距离阈值内；这里与ORB_SLAM2/3不同的是还添加了margin_far这个缩放因子，使得区间变的更加大了
        if (cam_to_lm_dist < min_cam_to_lm_dist || max_cam_to_lm_dist < cam_to_lm_dist) {
            continue;
        }

        // Compute the angle formed by the average vector of the 3D point observation,
        // and discard it if it is wider than the threshold value (60 degrees)
        const Vec3_t obs_mean_normal = lm->get_obs_mean_normal();

        // 夹角大于60度就不执行下面的操作了
        if (cam_to_lm_vec.dot(obs_mean_normal) < 0.5 * cam_to_lm_dist) {  // cos(theta) = 0.5 → theta = 60
            continue;
        }

        // Acquire keypoints in the cell where the reprojected 3D points exist
        const auto pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, keyfrm->orb_params_->num_levels_, keyfrm->orb_params_->log_scale_factor_);
        const auto indices = keyfrm->get_keypoints_in_cell(reproj(0), reproj(1), margin * keyfrm->orb_params_->scale_factors_.at(pred_scale_level));

        if (indices.empty()) {
            continue;
        }

        // Find a keypoint with the closest descriptor
        const auto lm_desc = lm->get_descriptor();

        unsigned int best_dist = MAX_HAMMING_DIST;
        int best_idx = -1;

        for (const auto idx : indices) {
            if (already_matched_idx_in_keyfrm.count(idx)) {
                continue;
            }
            const auto& undist_keypt = keyfrm->frm_obs_.undist_keypts_.at(idx);

            const auto scale_level = static_cast<unsigned int>(undist_keypt.octave);

            // TODO: shoud determine the scale with 'keyfrm-> get_keypts_in_cell ()'
            if (scale_level + 1 < pred_scale_level || pred_scale_level < scale_level) {  // scale_level只有等于pred_scale_level或者pred_scale_level - 1的时候才会往后面执行
                continue;
            }

            // 验证重投影误差，这里实际上是判断匹配的距离是否满足阈值
            if (do_reprojection_matching) {
                if (!keyfrm->frm_obs_.stereo_x_right_.empty() && keyfrm->frm_obs_.stereo_x_right_.at(idx) >= 0) {
                    // Compute reprojection error with 3 degrees of freedom if a stereo match exists
                    const auto e_x = reproj(0) - undist_keypt.pt.x;
                    const auto e_y = reproj(1) - undist_keypt.pt.y;
                    const auto e_x_right = x_right - keyfrm->frm_obs_.stereo_x_right_.at(idx);
                    const auto reproj_error_sq = e_x * e_x + e_y * e_y + e_x_right * e_x_right;

                    // n=3
                    constexpr float chi_sq_3D = 7.81473;
                    if (chi_sq_3D < reproj_error_sq * keyfrm->orb_params_->inv_level_sigma_sq_.at(scale_level)) {
                        continue;
                    }
                }
                else {
                    // Compute reprojection error with 2 degrees of freedom if a stereo match does not exist
                    const auto e_x = reproj(0) - undist_keypt.pt.x;
                    const auto e_y = reproj(1) - undist_keypt.pt.y;
                    const auto reproj_error_sq = e_x * e_x + e_y * e_y;

                    // n=2
                    constexpr float chi_sq_2D = 5.99146;
                    if (chi_sq_2D < reproj_error_sq * keyfrm->orb_params_->inv_level_sigma_sq_.at(scale_level)) {
                        continue;
                    }
                }
            }

            const auto& desc = keyfrm->frm_obs_.descriptors_.row(idx);

            const auto hamm_dist = compute_descriptor_distance_32(lm_desc, desc);

            if (hamm_dist < best_dist) {
                best_dist = hamm_dist;
                best_idx = idx;
            }
        }

        if (HAMMING_DIST_THR_LOW < best_dist) {
            continue;
        }

        already_matched_idx_in_keyfrm.insert(best_idx);
        auto lm_in_keyfrm = keyfrm->get_landmark(best_idx);
        if (lm_in_keyfrm) {
            // There is association between the 3D point and the keyframe
            // -> Duplication exists
            if (!lm_in_keyfrm->will_be_erased()) {
                duplicated_lms_in_keyfrm[lm] = lm_in_keyfrm;
            }
        }
        else {
            // There is no association between the 3D point and the keyframe
            // Add the observation information
            new_connections.emplace(best_idx, lm);
        }

        ++num_fused;
    }

    return num_fused;
}

// 模板的显示实例化
template unsigned int fuse::detect_duplication(const std::shared_ptr<data::keyframe>&,
                                               const Mat33_t&,
                                               const Vec3_t&,
                                               const std::vector<std::shared_ptr<data::landmark>>&,
                                               const float,
                                               std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>&,
                                               std::unordered_map<unsigned int, std::shared_ptr<data::landmark>>&,
                                               bool) const;
template unsigned int fuse::detect_duplication(const std::shared_ptr<data::keyframe>&,
                                               const Mat33_t&,
                                               const Vec3_t&,
                                               const id_ordered_set<std::shared_ptr<data::landmark>>&,
                                               const float,
                                               std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>&,
                                               std::unordered_map<unsigned int, std::shared_ptr<data::landmark>>&,
                                               bool) const;
template unsigned int fuse::detect_duplication(const std::shared_ptr<data::keyframe>&,
                                               const Mat33_t&,
                                               const Vec3_t&,
                                               const std::unordered_set<std::shared_ptr<data::landmark>>&,
                                               const float,
                                               std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>&,
                                               std::unordered_map<unsigned int, std::shared_ptr<data::landmark>>&,
                                               bool) const;
} // namespace match
} // namespace stella_vslam
