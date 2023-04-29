from src.stream_capture import StreamSaver
import cv2

if __name__ == '__main__':
    stream_saver = StreamSaver('video/1.mp4', 'save/videos/1/', fps_save=0.1, fps_update=20, shared_name='ndarray')
    import time
    start = time.time()
    while time.time() - start < 25:
        res, frame, timestamp = stream_saver.read()
        cv2.imshow('read', frame)
        res, frame = stream_saver.get(timestamp - 75)
        cv2.imshow('get', frame)
        cv2.waitKey(1)
    stream_saver.release()
