import cv2
import numpy as np


threshold1 = 10
threshold2 = 200    
rho = 1
theta = np.pi / 180
hough_threshold = 50
min_len = 50
max_gap = 10
min_pixels = 30


def detect_canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) 


def detect_hough(frame):
    edges = cv2.cvtColor(detect_canny(frame), cv2.COLOR_BGR2GRAY)
    output = frame.copy()
    lines = cv2.HoughLinesP(edges, rho, theta, hough_threshold,minLineLength=min_len, maxLineGap=max_gap)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return output


def detect_total_least_squares(frame):
    edges = cv2.cvtColor(detect_canny(frame), cv2.COLOR_BGR2GRAY)
    output = frame.copy()
    num_labels, labels = cv2.connectedComponents(edges, connectivity=8)

    for label in range(1, num_labels):
        y, x = np.where(labels == label)

        if len(x) < min_pixels:
            continue

        points = np.column_stack([x, y]).astype(np.float32)

        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

        pos = (points[:, 0] - x0) * vx + (points[:, 1] - y0) * vy
        point1 = (int(x0 + pos.min() * vx), int(y0 + pos.min() * vy))
        point2 = (int(x0 + pos.max() * vx), int(y0 + pos.max() * vy))

        cv2.line(output, point1, point2, (0, 0, 255), 2)

    return output

def compose_display(original, result1, result2, result3):
    panels = [original.copy(), result1.copy(), result2.copy(), result3.copy()]
    names = ["Original", "A: Canny", "B: Hough", "C: Least-Squares"]

    for panel, name in zip(panels, names):
        cv2.putText(panel, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    return np.vstack([top, bottom])


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("camera dont work")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (480, 360))

            result1 = detect_canny(frame)
            result2 = detect_hough(frame)
            result3 = detect_total_least_squares(frame)

            display = compose_display(frame, result1, result2, result3)
            cv2.imshow("camera", display)

            key = cv2.waitKey(1) 
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()