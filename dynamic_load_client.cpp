#include <opencv2/opencv.hpp>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include "falldetection_pipeline.hpp"



// ���Ͷ���
typedef FalldetectionPipeline* (*CreatePipelineFunc)(const std::string&, int);
typedef void (*DestroyPipelineFunc)(FalldetectionPipeline*);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "�÷�: " << argv[0] << " <��Ƶ·��>" << std::endl;
        return 1;
    }

    std::string video_path = argv[1];
    std::string config_path = "models.yaml";
    int dev_id = 0;

    // ���ض�̬��
    void* handle = dlopen("/path/to/lib/libfalldetection.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "�޷����ؿ�: " << dlerror() << std::endl;
        return 1;
    }

    // ��������
    CreatePipelineFunc create_pipeline = (CreatePipelineFunc)dlsym(handle, "create_falldetection_pipeline");
    DestroyPipelineFunc destroy_pipeline = (DestroyPipelineFunc)dlsym(handle, "destroy_falldetection_pipeline");
    if (!create_pipeline || !destroy_pipeline) {
        std::cerr << "�޷���������: " << dlerror() << std::endl;
        dlclose(handle);
        return 1;
    }

    try {
        // ���� pipeline ʵ��
        FalldetectionPipeline* pipeline = create_pipeline(config_path, dev_id);
        if (!pipeline) {
            throw std::runtime_error("�޷����� FalldetectionPipeline");
        }

        // ����Ƶ
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("�޷�����Ƶ: " + video_path);
        }

        // ��ʼ�������Ƶ
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::VideoWriter out("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
        if (!out.isOpened()) {
            throw std::runtime_error("�޷��������Ƶ: output.avi");
        }

        int frame_count = 0;
        while (true) {
            cv::Mat frame;
            if (!cap.read(frame)) {
                std::cout << "��Ƶ������ɣ��� " << frame_count << " ֡" << std::endl;
                break;
            }

            // ����
            ActionInferenceResult result = pipeline->inference(frame);
            frame_count++;

            // ��ӡ���
            std::cout << "֡ " << frame_count << ": ��⵽ " << result.online_targets.size() << " ��Ŀ��" << std::endl;
            for (size_t i = 0; i < result.labels.size(); ++i) {
                std::cout << "  Ŀ�� " << i << ": ��ǩ=" << result.labels[i]
                    << ", ����=" << result.probs[i]
                    << ", �ؼ�����=" << result.humans[i].size() << std::endl;
            }

            // ������ӻ�֡
            if (!result.visualized_frame.empty()) {
                out.write(result.visualized_frame);
            }
        }

        // �ͷ���Դ
        cap.release();
        out.release();
        pipeline->reset();
        destroy_pipeline(pipeline);
        dlclose(handle);
        std::cout << "�������" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "����: " << e.what() << std::endl;
        dlclose(handle);
        return 1;
    }

    return 0;
}