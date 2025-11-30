import cv2
import mediapipe as mp
import numpy as np
import time

# ===============================
# تنظیمات
# ===============================
DWELL_TIME = 2.1  # مدت موندن نگاه برای تایپ
typed_text = ""
gaze_start_time = None
current_key = None

#حساسیت چشم 

left_threshold = 0.4 
right_threshold = 0.6
smooth_frames = 10
hysteresis_count = 2

avg_history_h = [0.5]*smooth_frames
direction_history_h = {"LEFT":0,"CENTER":0,"RIGHT":0}

# ===============================
# MediaPipe Face Mesh
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

# ===============================
# Keyboard سه مرحله‌ای
# ===============================
keyboard_stage1 = [
    ["ABC","DEF","GHI"],
    ["JKL","MNO","PQR"],
    ["STU","VWX","YZ⌫"]
]

stage = 1  # 1: ستون، 2: ستون داخلی، 3: حرف
selected_col = 0
internal_layout = []
letter_options = []

# ===============================
# تابع رسم صفحه
# ===============================
def draw_stage(frame, layout, highlight_index):
    start_x, start_y = 50, 120
    w, h = 180, 90
    for row in range(len(layout)):
        for col in range(len(layout[row])):
            color = (0,255,0) if col==highlight_index else (255,255,255)
            x = start_x + col*w
            y = start_y + row*h
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            if layout[row][col]!="":
                cv2.putText(frame, layout[row][col], (x+20, y+60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# ===============================
# حلقه اصلی
# ===============================
horiz_dir = "CENTER"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    h,w,_ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    gaze_x_norm = 0.5

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        LEFT_EYE_POINTS = [33,133,159,145]
        RIGHT_EYE_POINTS = [362,263,386,374]

        def get_eye_roi(eye_points):
            xs = [face.landmark[p].x*w for p in eye_points]
            ys = [face.landmark[p].y*h for p in eye_points]
            x1,x2 = int(min(xs)), int(max(xs))
            y1,y2 = int(min(ys)), int(max(ys))
            return frame[y1:y2, x1:x2], x1, y1, x2, y2

        left_eye, lx1, ly1, lx2, ly2 = get_eye_roi(LEFT_EYE_POINTS)
        right_eye, rx1, ry1, rx2, ry2 = get_eye_roi(RIGHT_EYE_POINTS)

        left_gray = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        left_gray = cv2.GaussianBlur(left_gray,(5,5),0)
        right_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.GaussianBlur(right_gray,(5,5),0)

        thresh_val_left = max(30,int(np.mean(left_gray)*0.7))
        thresh_val_right = max(30,int(np.mean(right_gray)*0.7))

        _, left_thresh = cv2.threshold(left_gray, thresh_val_left, 255, cv2.THRESH_BINARY_INV)
        _, right_thresh = cv2.threshold(right_gray, thresh_val_right, 255, cv2.THRESH_BINARY_INV)

        left_coords = cv2.findNonZero(left_thresh)
        right_coords = cv2.findNonZero(right_thresh)

        if left_coords is not None and right_coords is not None:
            lx, ly = np.mean(left_coords, axis=0)[0]
            rx, ry = np.mean(right_coords, axis=0)[0]

            # نقطه مردمک
            cv2.circle(left_eye,(int(lx),int(ly)),3,(0,255,0),-1)
            cv2.circle(right_eye,(int(rx),int(ry)),3,(0,255,0),-1)

            lx_norm = lx/left_eye.shape[1]
            rx_norm = rx/right_eye.shape[1]
            avg_x = (lx_norm+rx_norm)/2

            avg_history_h.append(avg_x)
            if len(avg_history_h)>smooth_frames:
                avg_history_h.pop(0)
            smooth_h = np.median(avg_history_h)

            temp_dir_h = "CENTER"
            if smooth_h < left_threshold: temp_dir_h = "LEFT"
            elif smooth_h > right_threshold: temp_dir_h = "RIGHT"

            for key in direction_history_h:
                direction_history_h[key] = direction_history_h[key]+1 if key==temp_dir_h else 0
            if direction_history_h[temp_dir_h]>=hysteresis_count:
                horiz_dir = temp_dir_h

    # ===============================
    # مرحله ۱: ستون اصلی
    # ===============================
    if stage ==1:
        draw_stage(frame, keyboard_stage1, highlight_index={"LEFT":0,"CENTER":1,"RIGHT":2}[horiz_dir])
        if gaze_start_time is None:
            gaze_start_time = time.time()
            current_key = horiz_dir
        else:
            if horiz_dir==current_key:
                if time.time()-gaze_start_time>=DWELL_TIME:
                    selected_col = {"LEFT":0,"CENTER":1,"RIGHT":2}[horiz_dir]
                    # ساخت internal_layout
                    internal_layout=[]
                    for row in keyboard_stage1:
                        val = row[selected_col]
                        temp = [c for c in val]
                        while len(temp)<3: temp.append("")
                        internal_layout.append(temp)
                    stage =2
                    gaze_start_time=None
                    current_key=None
            else:
                gaze_start_time=time.time()
                current_key=horiz_dir

    # ===============================
    # مرحله ۲: ستون داخلی / ردیف
    # ===============================
    elif stage ==2:
        draw_stage(frame, internal_layout, highlight_index={"LEFT":0,"CENTER":1,"RIGHT":2}[horiz_dir])
        if gaze_start_time is None:
            gaze_start_time=time.time()
            current_key=horiz_dir
        else:
            if horiz_dir==current_key:
                if time.time()-gaze_start_time>=DWELL_TIME:
                    selected_internal_col={"LEFT":0,"CENTER":1,"RIGHT":2}[horiz_dir]
                    letter_options=[]
                    for row in internal_layout:
                        val=row[selected_internal_col]
                        if val!="":
                            letter_options.append(val)
                    stage=3
                    gaze_start_time=None
                    current_key=None
            else:
                gaze_start_time=time.time()
                current_key=horiz_dir

    # ===============================
    # مرحله ۳: انتخاب حرف با LEFT/CENTER/RIGHT
    # ===============================
    elif stage==3:
        draw_stage(frame, [letter_options], highlight_index={"LEFT":0,"CENTER":1,"RIGHT":2}[horiz_dir])
        if gaze_start_time is None:
            gaze_start_time=time.time()
            current_key=horiz_dir
        else:
            if horiz_dir==current_key:
                if time.time()-gaze_start_time>=DWELL_TIME:
                    idx = {"LEFT":0,"CENTER":1,"RIGHT":2}[horiz_dir]
                    if idx<len(letter_options):
                        val=letter_options[idx]
                        if val=="⌫":
                            typed_text=typed_text[:-1]
                        else:
                            typed_text+=val
                    stage=1
                    gaze_start_time=None
                    current_key=None
            else:
                gaze_start_time=time.time()
                current_key=horiz_dir

    # نمایش متن تایپ شده
    cv2.rectangle(frame,(30,30),(900,80),(0,0,0),-1)
    display_text = typed_text if len(typed_text)<60 else typed_text[-60:]
    cv2.putText(frame,display_text,(40,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    # نمایش جهت نگاه
    cv2.putText(frame,f"DIR: {horiz_dir}",(30,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Eye Keyboard",frame)
    if cv2.waitKey(1) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
