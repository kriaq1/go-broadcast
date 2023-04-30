from src.stream_capture import StreamSaver, available_camera_indexes_list, get_percentage_timestamp
import cv2

if __name__ == '__main__':
    # source = 'video/1.mp4'
    source = 0
    save_path = 'save/videos/1/'
    stream_saver = StreamSaver(source, save_path, fps_save=0.1, fps_update=20, shared_name='ndarray')
    import time

    start = time.time()
    while time.time() - start < 25:
        res, frame, timestamp = stream_saver.read()
        cv2.imshow('read', frame)
        res, frame = stream_saver.get(timestamp - 6000)
        cv2.imshow('get', frame)
        cv2.waitKey(1)
    print('0%: ', get_percentage_timestamp(save_path, 0))
    print('30%: ', get_percentage_timestamp(save_path, 40))
    print('60%: ', get_percentage_timestamp(save_path, 60))
    print('100%: ', get_percentage_timestamp(save_path, 100))
    stream_saver.release()

    first_list = available_camera_indexes_list()
    cap = cv2.VideoCapture(0)
    # cap.read()
    second_list = available_camera_indexes_list()
    cap.release()
    third_list = available_camera_indexes_list()

    print(first_list)
    print(second_list)
    print(third_list)
