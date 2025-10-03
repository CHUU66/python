# 目的：攝影機 → 用 MediaPipe FaceMesh 偵測臉 → 繪製出眼睛藍色框/嘴唇紅色框
# 離開鍵：按鍵盤 ESC

import cv2                      # OpenCV：處理攝影機與畫面顯示
import mediapipe as mp          

# 1) 打開預設攝影機（0 = 內建鏡頭；如果外接另一顆，可能要用 1 或 2）
cap = cv2.VideoCapture(0)

# 2) 建立 FaceMesh 偵測器
#    max_num_faces= *number*    ：表示偵測臉孔數的上限
#    refine_landmarks=*T/F*     ：True > 增加眼睛/嘴唇等細節點（更精細）
#    min_detection_confidence=*number0~1*   ：第一次偵測到臉的信心門檻
#    min_tracking_confidence=*number0~1*    ：已偵測到後，持續追蹤的門檻
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3)導入繪圖實用工具函數
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

while True:
    ok, frame = cap.read()   # 4) 讀取攝影機畫面
    if not ok:
        break                # 抓不到就離開（例如鏡頭被占用）
    
    frame = cv2.flip(frame, 1)            # 鏡像，互動直覺
    
    # 5) MediaPipe 要吃 RGB，所以先把 BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 6) FaceMesh 偵測
    res = face_mesh.process(rgb)

    # 7) 繪製臉部網格
    if res.multi_face_landmarks:
        h, w, _ = frame.shape   # 取得影像尺寸(height,width,channels(如 RGB 通道))
        for idx, face in enumerate(res.multi_face_landmarks):
            
            
            #嘴唇:
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face,
                connections=mp.solutions.face_mesh.FACEMESH_LIPS, 
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1)  # 紅色線
            )
            
            #左眼:
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face,
                connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYE, 
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1)  # 藍色線
            )
            #右眼:
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face,
                connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1)  # 藍色線
            )
            
    # 8) 顯示畫面視窗
    cv2.imshow("FaceMesh - Nose Tip (ESC to quit)", frame)
    # 9) 每 1ms 讀一次鍵盤，若按下 ESC (27) 就離開
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 10) 釋放攝影機並關閉所有視窗
cap.release()
cv2.destroyAllWindows()
