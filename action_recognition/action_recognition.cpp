
#include "action_recognition.hpp"
#include <stdexcept>

ActionRecognition::ActionRecognition(const std::string& model_path, int seg, int num_joint,
    int num_classes, int channels, int dev_id)
    : seg_(seg), num_joint_(num_joint), num_classes_(num_classes), channels_(channels),
    labels_({ "fall", "normal" }) {
    // ��ʼ�� Sophgo BMNN �����������
    auto bm_handle = std::make_shared<BMNNHandle>(dev_id);
    bm_ctx_ = std::make_shared<BMNNContext>(bm_handle, model_path.c_str());
    network_ = std::make_shared<BMNNNetwork>(bm_ctx_->bmrt(), bm_ctx_->network_name(0));

    // ��ȡ����������״
    input_shape_ = *network_->inputTensor(0)->get_shape();
    output_shape_ = *network_->outputTensor(0)->get_shape();

    // ��֤������״ [1, seg, num_joint * channels]
    if (input_shape_.dims[0] != 1 || input_shape_.dims[1] != seg ||
        input_shape_.dims[2] != num_joint * channels) {
        throw std::runtime_error("Invalid SGN input shape, expected [1, " +
            std::to_string(seg) + ", " +
            std::to_string(num_joint * channels) + "]");
    }

    // ��֤�����״ [1, num_classes]
    if (output_shape_.dims[0] != 1 || output_shape_.dims[1] != num_classes) {
        throw std::runtime_error("Invalid SGN output shape, expected [1, " +
            std::to_string(num_classes) + "]");
    }

    // ��֤ labels_ �Ĵ�С�� num_classes һ��
    if (labels_.size() != static_cast<size_t>(num_classes)) {
        throw std::runtime_error("Labels size (" + std::to_string(labels_.size()) +
            ") does not match num_classes (" + std::to_string(num_classes) + ")");
    }
}

ActionRecognition::~ActionRecognition() {
    // �ͷ���Դ
    //if (bm_ctx_ && bm_ctx_->handle()) {
    //    // ȷ�����������������豸�ڴ����ͷ�
    //    for (int i = 0; i < network_->get_num_inputs(); ++i) {
    //        const bm_device_mem_t* input_mem = network_->inputTensor(i)->get_device_mem();
    //        if (input_mem && bm_mem_get_device_addr(*input_mem) != 0) {
    //            bm_free_device(bm_ctx_->handle(), const_cast<bm_device_mem_t>(*input_mem));
    //        }
    //    }
    //    // ȷ����������������豸�ڴ����ͷ�
    //    for (int i = 0; i < network_->get_num_outputs(); ++i) {
    //        const bm_device_mem_t* output_mem = network_->outputTensor(i)->get_device_mem();
    //        if (output_mem && bm_mem_get_device_addr(*output_mem) != 0) {
    //            bm_free_device(bm_ctx_->handle(), const_cast<bm_device_mem_t>(*output_mem));
    //        }
    //    }
    //}
    // bm_ctx_ �� network_ �������� shared_ptr �Զ�����
}

std::pair<std::string, float> ActionRecognition::infer(const std::vector<std::vector<cv::Point2f>>& frames_buffer) {
    if (frames_buffer.size() < static_cast<size_t>(seg_)) {
        return { "Tracking", 0.0f };
    }

    // ׼����������
    std::vector<float> input_data(seg_ * num_joint_ * channels_);
    for (int t = 0; t < seg_; ++t) {
        for (int j = 0; j < num_joint_; ++j) {
            input_data[t * num_joint_ * channels_ + j * channels_] = frames_buffer[t][j].x;
            input_data[t * num_joint_ * channels_ + j * channels_ + 1] = frames_buffer[t][j].y;
        }
    }

    // �����豸�ڴ沢��������
    bm_device_mem_t input_mem;
    bm_malloc_device_byte(bm_ctx_->handle(), &input_mem, input_shape_.dims[1] * input_shape_.dims[2] * sizeof(float));
    bm_memcpy_s2d(bm_ctx_->handle(), input_mem, input_data.data());
    network_->inputTensor(0)->set_device_mem(&input_mem);

    // ִ��ǰ������
    network_->forward();

    // ��ȡ�������
    std::vector<float> output_data(num_classes_);
    bm_device_mem_t output_mem = *network_->outputTensor(0)->get_device_mem();
    bm_memcpy_d2s(bm_ctx_->handle(), output_data.data(), output_mem);

    // �ҵ������ʵ����
    float max_prob = output_data[0];
    int max_idx = 0;
    for (int i = 1; i < num_classes_; ++i) {
        if (output_data[i] > max_prob) {
            max_prob = output_data[i];
            max_idx = i;
        }
    }

    // ʹ�� labels_ ��Ա������ȡ��ǩ
    std::string label = labels_[max_idx];
    float prob = max_prob / (output_data[0] + output_data[1]);

    // �ͷ������豸�ڴ�
    bm_free_device(bm_ctx_->handle(), input_mem);

    return { label, prob };
}
