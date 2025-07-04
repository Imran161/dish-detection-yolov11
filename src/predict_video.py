import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

DEVICE = 'cuda:0'

def process_video(model_path, input_video_path, output_video_path):
    model = YOLO(model_path)
    model.to(DEVICE) 
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {input_video_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Начинаем обработку видео: {input_video_path}")
    print(f"Размер кадра: {frame_width}x{frame_height}, FPS: {fps:.2f}, Всего кадров: {total_frames}")
    
    pbar = tqdm(total=total_frames, desc="Обработка видео")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.8, verbose=False)
        
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)
        
        pbar.update(1)
    
    cap.release()
    out.release()
    pbar.close()
    
    print(f"Видео сохранено: {output_video_path}")

if __name__ == "__main__":
    model_path = "/home/imran-nasyrov/test_y/outputs/train/default16/weights/best.pt"
    input_video_path = "/home/imran-nasyrov/test_y/dish-detection-3/4_1.MOV"
    output_video_path = "/home/imran-nasyrov/test_y/4_1_annotated.mp4"
    process_video(model_path, input_video_path, output_video_path)