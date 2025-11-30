import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# تنظیمات حساسیت
left_threshold = 0.4
right_threshold = 0.6
smooth_frames = 10  # تعداد فریم برای median smoothing
hysteresis_count = 2  # تعداد فریم پشت سر هم برای تایید direction

avg_history = [0.5] * smooth_frames
direction_history = {"LEFT":0, "CENTER":0, "RIGHT":0}
direction = "CENTER"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        LEFT_EYE_POINTS = [33, 133, 159, 145]
        RIGHT_EYE_POINTS = [362, 263, 386, 374]

        def get_eye_roi(eye_points):
            xs = [face.landmark[p].x*w for p in eye_points]
            ys = [face.landmark[p].y*h for p in eye_points]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            return frame[y1:y2, x1:x2], x1, y1, x2, y2

        left_eye, lx1, ly1, lx2, ly2 = get_eye_roi(LEFT_EYE_POINTS)
        right_eye, rx1, ry1, rx2, ry2 = get_eye_roi(RIGHT_EYE_POINTS)

        # Gaussian blur برای کاهش نویز
        left_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        left_gray = cv2.GaussianBlur(left_gray, (5,5), 0)

        right_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.GaussianBlur(right_gray, (5,5), 0)

        # threshold دینامیک
        thresh_val_left = max(30, int(np.mean(left_gray)*0.7))
        thresh_val_right = max(30, int(np.mean(right_gray)*0.7))

        _, left_thresh = cv2.threshold(left_gray, thresh_val_left, 255, cv2.THRESH_BINARY_INV)
        _, right_thresh = cv2.threshold(right_gray, thresh_val_right, 255, cv2.THRESH_BINARY_INV)

        left_coords = cv2.findNonZero(left_thresh)
        right_coords = cv2.findNonZero(right_thresh)

        if left_coords is not None and right_coords is not None:
            lx, ly = np.mean(left_coords, axis=0)[0]
            rx, ry = np.mean(right_coords, axis=0)[0]

            cv2.circle(left_eye, (int(lx), int(ly)), 3, (0,255,0), -1)
            cv2.circle(right_eye, (int(rx), int(ry)), 3, (0,255,0), -1)

            # موقعیت نسبی مردمک نسبت به چشم خودش
            lx_norm = lx / left_eye.shape[1]
            rx_norm = rx / right_eye.shape[1]
            avg_norm = (lx_norm + rx_norm) / 2

            # median smoothing
            avg_history.append(avg_norm)
            if len(avg_history) > smooth_frames:
                avg_history.pop(0)
            smooth_avg = np.median(avg_history)

            # تشخیص direction موقت
            temp_direction = "CENTER"
            if smooth_avg < left_threshold:
                temp_direction = "LEFT"
            elif smooth_avg > right_threshold:
                temp_direction = "RIGHT"

            # hysteresis
            for key in direction_history:
                if key == temp_direction:
                    direction_history[key] +=1
                else:
                    direction_history[key] = 0

            if direction_history[temp_direction] >= hysteresis_count:
                direction = temp_direction

        cv2.imshow("Left Eye", left_eye)
        cv2.imshow("Right Eye", right_eye)

    cv2.putText(frame, direction, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,0,255), 2)

    cv2.imshow("Eye Direction", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
