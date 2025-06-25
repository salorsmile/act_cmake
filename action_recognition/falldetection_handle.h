// falldetection_handle.h
#ifndef FALLDETECTION_HANDLE_H
#define FALLDETECTION_HANDLE_H

#ifdef __cplusplus
extern "C" {
#endif

    // ���嵼����
#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

// ��͸���������
    typedef void* FalldetectionHandle;

    // ��ʾ��������Ŀ�����Ϣ��C �汾��
    typedef struct {
        int track_id;         // ����ID
        int state;            // ����״̬
        float* tlbr;          // �߽�� (top-left-bottom-right)������Ϊ4
        int frame_id;         // ��ǰ֡ID
        int tracklet_len;     // ���ٳ���ʱ��
        int start_frame;      // ���ٿ�ʼ֡
        float score;          // ���÷�
        int class_id;         // ���ID
    } CTrackEntry;

    // ��ʾ������Ϣ�������������Ŀ�꣨C �汾��
    typedef struct {
        CTrackEntry* targets;  // ����Ŀ������
        int target_count;      // Ŀ������
    } CTrackInfo;

    // ��ʾ�ؼ�����Ϣ
    typedef struct {
        float x;
        float y;
    } Point2f;

    // ��ʾһ���˵Ĺؼ��㼯��
    typedef struct {
        Point2f* points;      // �ؼ�������
        int point_count;      // �ؼ�������
    } KeypointSet;

    // ��ʾ��������C �汾��
    typedef struct {
        unsigned char* visualized_frame_data; // ���ӻ�֡��ͼ�����ݣ�BGR��ʽ��
        int frame_width;                      // ֡���
        int frame_height;                     // ֡�߶�
        int frame_channels;                   // ֡ͨ����
        KeypointSet* humans;                  // �˵Ĺؼ�������
        int human_count;                      // ����
        CTrackInfo online_targets;            // ������Ϣ
        char** labels;                        // ������ǩ����
        float* probs;                         // ������������
        int label_count;                      // ��ǩ����
    } CActionInferenceResult;

    // ������
    EXPORT_API FalldetectionHandle falldetection_create(const char* config_path, int dev_id);

    // ���پ��
    EXPORT_API void falldetection_destroy(FalldetectionHandle handle);

    // ִ���������� cv::Mat ��Ϊ void*��
    EXPORT_API int falldetection_inference(FalldetectionHandle handle,
        void* image,
        CActionInferenceResult* result);

    // �ͷ����������ڴ�
    EXPORT_API void falldetection_free_result(CActionInferenceResult* result);

    // ����״̬
    EXPORT_API void falldetection_reset(FalldetectionHandle handle);

#ifdef __cplusplus
}
#endif

#endif // FALLDETECTION_HANDLE_H