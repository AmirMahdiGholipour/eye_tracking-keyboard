import cv2

keyboard = [
    [["A","B","C"], ["D","E","F"], ["G","H","I"]],
    [["J","K","L"], ["M","N","O"], ["P","Q","R"]],
    [["S","T","U"], ["V","W","X"], ["Y","Z","⌫"]]
]

typed_text = ""
selected_col = 0
selected_row = 0
mode = "CELL"
current_group = None
letter_index = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # نمایش متن تایپ‌شده
    cv2.rectangle(frame, (30, 30), (900, 80), (0,0,0), -1)
    display_text = typed_text if len(typed_text) < 60 else typed_text[-60:]
    cv2.putText(frame, display_text, (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    key = cv2.waitKey(1)

    if mode == "CELL":
        start_x = 50
        start_y = 120
        w = 140
        h = 90

        # رسم خانه‌ها
        for row in range(3):
            for col in range(3):
                x = start_x + col * w
                y = start_y + row * h

                if row == selected_row and col == selected_col:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), -1)
                    color = (0,0,0)
                else:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)
                    color = (0,255,0)

                txt = " ".join(keyboard[row][col])
                cv2.putText(frame, txt, (x+10, y+60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # اسپیس
        space_y = start_y + 3*h + 20
        sw = w*3
        if selected_row == 3:
            cv2.rectangle(frame, (start_x, space_y), (start_x+sw, space_y+70), (0,255,0), -1)
            sc = (0,0,0)
        else:
            cv2.rectangle(frame, (start_x, space_y), (start_x+sw, space_y+70), (255,255,255), 2)
            sc = (0,255,0)

        cv2.putText(frame, "SPACE", (start_x + sw//2 - 60, space_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, sc, 2)

        # حرکت WASD
        if key == ord('a') and selected_row < 3: selected_col = max(0, selected_col-1)
        if key == ord('d') and selected_row < 3: selected_col = min(2, selected_col+1)
        if key == ord('w'): selected_row = max(0, selected_row-1)
        if key == ord('s'): selected_row = min(3, selected_row+1)

        # انتخاب
        if key == 13:
            if selected_row == 3:
                typed_text += " "
            else:
                current_group = keyboard[selected_row][selected_col]
                letter_index = 0
                mode = "LETTER"

    else:
        # حالت انتخاب حرف
        x = 200
        y = 200
        gap = 180

        for i in range(3):
            color = (0,255,0) if i == letter_index else (255,255,255)
            cv2.rectangle(frame, (x + i*gap, y), (x+120 + i*gap, y+100), color, 2)
            cv2.putText(frame, current_group[i],
                        (x + i*gap + 30, y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        color, 3)

        # انتخاب با A / D
        if key == ord('a'):
            letter_index = max(0, letter_index-1)
        if key == ord('d'):
            letter_index = min(2, letter_index+1)

        # تایید
        if key == 13:
            val = current_group[letter_index]
            if val == "⌫":
                typed_text = typed_text[:-1]
            else:
                typed_text += val

            mode = "CELL"

    cv2.imshow("Eye Keyboard", frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
