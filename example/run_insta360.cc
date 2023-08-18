#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#elif USE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif

#include "stella_vslam/type.h"
#include "stella_vslam/system.h"
#include "stella_vslam/config.h"
#include "stella_vslam/camera/base.h"
#include "stella_vslam/util/yaml.h"
#include "stella_vslam/util/image.h"
#include "stella_vslam/data/landmark.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <numeric>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>
#include <iomanip>

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

#ifdef USE_STACK_TRACE_LOGGER
#include <backward.hpp>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

cv::Mat save_match_info(const std::shared_ptr<stella_vslam::data::keyframe> &keyfrm, cv::Mat &candidate_image, const stella_vslam::data::frame &frame,
                        cv::Mat &curr_image, unsigned int rows) {
//    for (auto &kpt: frame.frm_obs_.undist_keypts_)
//        cv::circle(curr_image, kpt.pt, 5, cv::Scalar(255, 0, 0), 2, 4, 0);
//
//    for (auto &kpt: keyfrm->frm_obs_.undist_keypts_) {
//        cv::circle(candidate_image, kpt.pt, 5, cv::Scalar(255, 0, 0), 2, 4, 0);
//    }

    for (auto &lm: keyfrm->get_landmarks()) {
        if (!lm)
            continue;

        const auto idx = lm->get_index_in_keyframe(keyfrm);
        if (idx == -1)
            continue;

        auto kpt = keyfrm->frm_obs_.undist_keypts_[idx];
        cv::circle(candidate_image, kpt.pt, 5, cv::Scalar(255, 0, 0), 2, 4, 0);
    }


    cv::Mat combine_image;
    cv::vconcat(curr_image, candidate_image, combine_image);

    std::vector<std::shared_ptr<stella_vslam::data::landmark>> matched_landmarks = keyfrm->get_matched_landmarks();
    for (unsigned int idx = 0; idx < matched_landmarks.size(); ++idx) {
        auto& lm = matched_landmarks.at(idx);
        if (!lm)
            continue;

        auto kpt1 = frame.frm_obs_.undist_keypts_[idx];
        const auto idx_2 = lm->get_index_in_keyframe(keyfrm);
        if (idx_2 == -1)
            continue;

        auto kpt2 = keyfrm->frm_obs_.undist_keypts_[idx_2];
        cv::Point2f pt2 = cv::Point2f(kpt2.pt.x, kpt2.pt.y + rows);
        cv::circle(combine_image, kpt1.pt, 5, cv::Scalar(0, 255, 0), 2, 4, 0);
        cv::circle(combine_image, pt2, 5, cv::Scalar(0, 255, 255), 2, 4, 0);
        cv::line(combine_image, kpt1.pt, pt2, cv::Scalar(0, 255, 0));
    }

    return combine_image;
}

void mono_tracking(const std::shared_ptr<stella_vslam::system>& slam,
                   const std::shared_ptr<stella_vslam::config>& cfg,
                   const std::string& video_file_path,
                   const std::string& mask_img_path,
                   const unsigned int frame_skip,
                   const unsigned int start_time,
                   const bool no_sleep,
                   const bool wait_loop_ba,
                   const bool auto_term,
                   const std::string& eval_log_dir,
                   const std::string& map_db_path,
                   const double start_timestamp,
                   const unsigned int start_frame,
                   const unsigned int end_frame,
                   const bool is_reloc,
                   const std::string &track_pose_file = "",
                   const std::string &map_pose_file = "",
                   const std::string &map_image_file = "",
                   const std::string &reloc_candidate_file = "") {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(
        stella_vslam::util::yaml_optional_ref(cfg->yaml_node_, "PangolinViewer"), slam, slam->get_frame_publisher(), slam->get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(
        stella_vslam::util::yaml_optional_ref(cfg->yaml_node_, "SocketPublisher"), slam, slam->get_frame_publisher(), slam->get_map_publisher());
#endif

    auto video = cv::VideoCapture(video_file_path, cv::CAP_FFMPEG);
    if (!video.isOpened()) {
        std::cerr << "Unable to open the video." << std::endl;
        return;
    }
    video.set(0, start_time);
    video.set(cv::CAP_PROP_POS_FRAMES, start_frame);

    std::vector<double> track_times;

    cv::Mat frame;

    unsigned int num_frame = 0;
    double timestamp = start_timestamp;
    // 有可能不是从第一帧开始的，记录这个值，方便从时间戳上面得到帧号
    if (abs(start_timestamp) < 1e-6) {
        timestamp += start_frame / slam->get_camera()->fps_;
    }

    // 保存重定位相关的信息
    int reloc_total = 0, reloc_succ = 0;

    // save track pose, generally use to eval reloc
    std::ofstream ofs_track_pose;
    if (!track_pose_file.empty()) {
        ofs_track_pose.open(track_pose_file);
        if (!ofs_track_pose.is_open()) {
            std::cerr << "Unable to open the reloc file: " << track_pose_file << std::endl;
            return;
        }

        ofs_track_pose << std::fixed << std::setprecision(6);
        ofs_track_pose << "# frame_id(start 0) x y z qx qy qz qw" << std::endl;
    }

    bool is_not_end = true;
    // run the slam in another thread
    std::thread thread([&]() {
      while (is_not_end) {
          // wait until the loop BA is finished
          if (wait_loop_ba) {
              while (slam->loop_BA_is_running() || !slam->mapping_module_is_enabled()) {
                  std::this_thread::sleep_for(std::chrono::milliseconds(100));
              }
          }

          is_not_end = video.read(frame);

          const auto tp_1 = std::chrono::steady_clock::now();
          if (end_frame != 0 && num_frame + start_frame > end_frame)
              break;

          // frame by frame reloc
          if (is_reloc) {
              if (num_frame % frame_skip == 0)
                  std::cout << "Frame by frame reloc, current frame num: " << num_frame << std::endl;
              slam->set_track_state(stella_vslam::tracker_state_t::Lost);
          }

          // 全景图像有时候会是纯色的
          if (!frame.empty() && (num_frame % frame_skip == 0) && !stella_vslam::util::is_pure_color(frame)) {
              // input the current frame and estimate the camera pose: the pose will be nullptr
              std::shared_ptr<stella_vslam::Mat44_t> Twc = slam->feed_monocular_frame(frame, timestamp, mask);  // track result, not map result, only use to reloc analysis

#if 0
              if (num_frame % 20 == 0)
                cv::imwrite("/home/xiongchao/图片/images/" + std::to_string(num_frame) + ".jpg", frame);
#endif

              if (!track_pose_file.empty() && Twc) {  // judge nullptr
                  Eigen::Quaterniond q(Twc->block<3, 3>(0, 0));
                  Eigen::Vector3d t = Twc->block<3, 1>(0, 3);
                  ofs_track_pose << num_frame << " "
                                 << t(0, 0) << " " << t(1, 0) << " " << t(2, 0) << " "
                                 << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                                 << std::endl;
              }

              if (is_reloc)
                reloc_total++;
              // stat reloc succ frame
              if (is_reloc && slam->get_track_state() == stella_vslam::tracker_state_t::Tracking)
                  reloc_succ++;

              // record current frame and candidate keyframes, then combine them
              if (is_reloc && !map_image_file.empty() && !reloc_candidate_file.empty() && slam->get_track_state() == stella_vslam::tracker_state_t::Tracking) {
                  std::vector<std::shared_ptr<stella_vslam::data::keyframe>> candidate_keyframes = slam->get_tracker()->get_relocalizer().get_candidate_keyframes();
                  std::cout << "candidate_keyframes.size = " << candidate_keyframes.size() << std::endl;

#if 0
                  std::shared_ptr<stella_vslam::data::keyframe> success_keyfrm = slam->get_tracker()->get_relocalizer().get_success_keyframe();
                  int frame_count = round(success_keyfrm->timestamp_ * slam->get_camera()->fps_);
                  std::string candidate_image_path = map_image_file + "/" + std::to_string(frame_count) + ".jpg";
                  cv::Mat candidate_image = cv::imread(candidate_image_path);
#if 0
                  cv::imwrite("/home/xiongchao/视频/image3/source-" + std::to_string(num_frame) + ".jpg", frame);
                  cv::imwrite("/home/xiongchao/视频/image4/query-" + std::to_string(frame_count) + ".jpg", candidate_image);
#endif
                  cv::Mat combine_image = save_match_info(success_keyfrm, candidate_image, slam->get_tracker()->curr_frm_, frame, slam->get_camera()->rows_);
                  std::string combine_image_name = reloc_candidate_file + "/" + std::to_string(num_frame) + "-" + std::to_string(frame_count) + ".jpg";
                  cv::imwrite(combine_image_name, combine_image);
#endif

#if 0
                  for (auto &candidate_keyframe: candidate_keyframes) {
                      int frame_count = round(candidate_keyframe->timestamp_ * slam->get_camera()->fps_);
                      std::string candidate_image_path = map_image_file + "/" + std::to_string(frame_count) + ".jpg";
                      cv::Mat candidate_image = cv::imread(candidate_image_path);
#if 0
                      cv::imwrite("/home/xiongchao/视频/image1/source-" + std::to_string(num_frame) + ".jpg", frame);
                      cv::imwrite("/home/xiongchao/视频/image2/query-" + std::to_string(frame_count) + ".jpg", candidate_image);
#endif

                      cv::Mat combine_image = save_match_info(candidate_keyframe, candidate_image, slam->get_tracker()->curr_frm_, frame, slam->get_camera()->rows_);
                      std::string combine_image_name = reloc_candidate_file + "/" + std::to_string(num_frame) + "-" + std::to_string(frame_count) + ".jpg";
                      cv::imwrite(combine_image_name, combine_image);
                  }
#endif
              }
          }

          const auto tp_2 = std::chrono::steady_clock::now();

          const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
          if (num_frame % frame_skip == 0) {
              track_times.push_back(track_time);
          }

          // wait until the timestamp of the next frame
          if (!no_sleep) {
              // 经过测试，绝大部分时候都是负数
              const auto wait_time = 1.0 / slam->get_camera()->fps_ - track_time;
              if (0.0 < wait_time) {
                  std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(wait_time * 1e6)));
              }
          }

          timestamp += 1.0 / slam->get_camera()->fps_;
          ++num_frame;

          // check if the termination of slam system is requested or not
          if (slam->terminate_is_requested()) {
              break;
          }
      }

      // wait until the loop BA is finished
      while (slam->loop_BA_is_running()) {
          std::this_thread::sleep_for(std::chrono::microseconds(5000));
      }

      // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
      if (auto_term) {
            viewer.request_terminate();
        }
#elif USE_SOCKET_PUBLISHER
      if (auto_term) {
            publisher.request_terminate();
        }
#endif
    });

    // run the viewer in the current thread
#ifdef USE_PANGOLIN_VIEWER
    viewer.run();
#elif USE_SOCKET_PUBLISHER
    publisher.run();
#endif

    thread.join();

    // shutdown the slam process
    slam->shutdown();

    if (!eval_log_dir.empty()) {
        // output the trajectories for evaluation
        slam->save_frame_trajectory(eval_log_dir + "/frame_trajectory.txt", "TUM");
        slam->save_keyframe_trajectory(eval_log_dir + "/keyframe_trajectory.txt", "TUM");
        // output the tracking times for evaluation
        std::ofstream ofs(eval_log_dir + "/track_times.txt", std::ios::out);
        if (ofs.is_open()) {
            for (const auto track_time : track_times) {
                ofs << track_time << std::endl;
            }
            ofs.close();
        }
    }

    if (!map_db_path.empty()) {
        // output the map database
        slam->save_map_database(map_db_path);
    }

    if (!map_pose_file.empty())
        slam->save_pose_txt(map_pose_file);


    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;

    // stat reloc result
    if (is_reloc && reloc_total > 0)
        std::cout << "reloc stat: reloc_total = " << reloc_total << ", reloc_succ = " << reloc_succ << ", succ_ratio = " << double(reloc_succ) / reloc_total << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    backward::SignalHandling sh;
#endif

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto video_file_path = op.add<popl::Value<std::string>>("m", "video", "video file path");
    auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
    auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
    auto frame_skip = op.add<popl::Value<unsigned int>>("", "frame-skip", "interval of frame skip", 1);
    auto start_time = op.add<popl::Value<unsigned int>>("s", "start-time", "time to start playing [milli seconds]", 0);
    auto no_sleep = op.add<popl::Switch>("", "no-sleep", "not wait for next frame in real time");
    auto wait_loop_ba = op.add<popl::Switch>("", "wait-loop-ba", "wait until the loop BA is finished");
    auto auto_term = op.add<popl::Switch>("", "auto-term", "automatically terminate the viewer");
    auto log_level = op.add<popl::Value<std::string>>("", "log-level", "log level", "info");
    auto eval_log_dir = op.add<popl::Value<std::string>>("", "eval-log-dir", "store trajectory and tracking times at this path (Specify the directory where it exists.)", "");
    auto map_db_path_in = op.add<popl::Value<std::string>>("i", "map-db-in", "load a map from this path", "");
    auto map_db_path_out = op.add<popl::Value<std::string>>("o", "map-db-out", "store a map database at this path after slam", "");
    auto disable_mapping = op.add<popl::Switch>("", "disable-mapping", "disable mapping");
    auto start_timestamp = op.add<popl::Value<double>>("t", "start-timestamp", "timestamp of the start of the video capture");
    auto start_frame = op.add<popl::Value<unsigned int>>("", "start-frame", "interval of frame skip", 0);
    auto end_frame = op.add<popl::Value<unsigned int>>("", "end-frame", "interval of frame skip", 0);
    auto is_reloc = op.add<popl::Switch>("", "is-reloc", "every frame exec reloc");
    auto track_pose_file = op.add<popl::Value<std::string>>("", "track-pose-file", "store reloc result", "");
    auto map_pose_file = op.add<popl::Value<std::string>>("", "map-pose-file", "store keyframe pose in map", "");
    auto map_image_file = op.add<popl::Value<std::string>>("", "map-image-file", "search candidate keyframe in video when reloc", "");
    auto reloc_candidate_file = op.add<popl::Value<std::string>>("", "reloc-candidate-file", "store current frame and candidate keyframe when reloc", "");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!op.unknown_options().empty()) {
        for (const auto& unknown_option : op.unknown_options()) {
            std::cerr << "unknown_options: " << unknown_option << std::endl;
        }
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !video_file_path->is_set() || !config_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    spdlog::set_level(spdlog::level::from_str(log_level->value()));

    // load configuration
    std::shared_ptr<stella_vslam::config> cfg;
    try {
        cfg = std::make_shared<stella_vslam::config>(config_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    // You cannot get timestamps of images with this input format.
    // It is recommended to specify the timestamp when the video recording was started in Unix time.
    // If not specified, the current system time is used instead.
    double timestamp = 0.0;
    if (!start_timestamp->is_set()) {
        std::cerr << "--start-timestamp is not set. using system timestamp." << std::endl;
        if (no_sleep->is_set()) {
            std::cerr << "If --no-sleep is set without --start-timestamp, timestamps may overlap between multiple runs." << std::endl;
        }
        std::chrono::system_clock::time_point start_time_system = std::chrono::system_clock::now();
        timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(start_time_system.time_since_epoch()).count();
    }
    else {
        timestamp = start_timestamp->value();
    }


    // build a slam system
    auto slam = std::make_shared<stella_vslam::system>(cfg, vocab_file_path->value());

    bool need_initialize = true;
    if (map_db_path_in->is_set()) {
        need_initialize = false;
        const auto path = fs::path(map_db_path_in->value());
        if (path.extension() == ".yaml") {
            YAML::Node node = YAML::LoadFile(path);
            for (const auto& map_path : node["maps"].as<std::vector<std::string>>()) {
                slam->load_map_database(path.parent_path() / map_path);
            }
        }
        else {
            // load the prebuilt map
            slam->load_map_database(path);
        }
    }

    // is_stopped_keyframe_insertion_在这个函数中被设置，因此，只要有加载地图，就不会在生成新的关键帧了
    slam->startup(need_initialize);
    if (disable_mapping->is_set()) {
        slam->disable_mapping_module();
    }

    // run tracking
    if (slam->get_camera()->setup_type_ == stella_vslam::camera::setup_type_t::Monocular) {
        mono_tracking(slam,
                      cfg,
                      video_file_path->value(),
                      mask_img_path->value(),
                      frame_skip->value(),
                      start_time->value(),
                      no_sleep->is_set(),
                      wait_loop_ba->is_set(),
                      auto_term->is_set(),
                      eval_log_dir->value(),
                      map_db_path_out->value(),
                      timestamp,
                      start_frame->value(),
                      end_frame->value(),
                      is_reloc->is_set(),
                      track_pose_file->value(),
                      map_pose_file->value(),
                      map_image_file->value(),
                      reloc_candidate_file->value());
    }
    else {
        throw std::runtime_error("Invalid setup type: " + slam->get_camera()->get_setup_type_string());
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
}
