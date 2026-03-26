import argparse
import time
import logging
import cv2
from ultralytics import YOLO


def draw_boxes(frame, results, names, exclude=None):
    """Draw detected boxes on frame, skipping any class in `exclude`."""
    exclude_set = set(exclude or [])
    for box in results.boxes:
        cls = int(box.cls[0])
        class_name = names[cls]
        if class_name in exclude_set:
            continue
        xyxy = box.xyxy[0]
        x1, y1, x2, y2 = map(int, xyxy)
        conf = float(box.conf[0])
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def run_detector(source=0, model_path='yolov8n.pt', conf=0.35, iou=0.45, save_path=None, show_fps=True, exclude=None):
    logging.info('Lade Modell: %s', model_path)
    model = YOLO(model_path)

    # default to excluding persons if nothing else provided
    if exclude is None:
        exclude = ['person']
    # Normalize exclude to a set of class names
    exclude_set = set(exclude or [])

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error('Kamera/Quelle %s kann nicht geöffnet werden', source)
        return

    out_writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    prev_time = time.time()
    frame_count = 0

    logging.info('Drücke q zum Beenden')
    first_attempt = True
    while True:
        ret, frame = cap.read()
        if not ret:
            if first_attempt:
                logging.error('Kein Bild von der Quelle erhalten. Bitte Kameraindex prüfen oder sicherstellen, dass keine andere Anwendung den Zugriff blockiert.')
            else:
                logging.info('Kein Frame erhalten, beende')
            break
        first_attempt = False

        results = model(frame, conf=conf, iou=iou)[0]

        draw_boxes(frame, results, model.names, exclude=exclude_set)

        if show_fps:
            frame_count += 1
            if frame_count >= 10:
                now = time.time()
                fps = frame_count / (now - prev_time)
                prev_time = now
                frame_count = 0
            else:
                fps = None
            if fps:
                cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('YOLOv8 Echtzeit-Objekterkennung', frame)

        if out_writer is not None:
            out_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 Echtzeit-Objekterkennung')
    parser.add_argument('--source', type=str, default='0', help='Kameraindex oder Pfad zu Video-Datei (default 0)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pfad zum YOLOv8 Modell')
    parser.add_argument('--conf', type=float, default=0.35, help='Konfidenz-Threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU für NMS')
    parser.add_argument('--save', type=str, default=None, help='Optional: Pfad um Ausgabevideo zu speichern (mp4)')
    parser.add_argument('--no-fps', action='store_true', help='FPS-Anzeige deaktivieren')
    parser.add_argument('--exclude', type=str, default=None, help='Komma-getrennte Klassennamen ausschließen, z.B. "dog,cat" (Standard: person wird ausgeschlossen)')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    args = parse_args()
    # Allow numeric camera index
    source = int(args.source) if args.source.isdigit() else args.source
    exclude_list = None
    if args.exclude:
        exclude_list = [s.strip() for s in args.exclude.split(',') if s.strip()]
        logging.info('Ausschlussliste: %s', exclude_list)
    run_detector(source=source, model_path=args.model, conf=args.conf, iou=args.iou, save_path=args.save, show_fps=not args.no_fps, exclude=exclude_list)
