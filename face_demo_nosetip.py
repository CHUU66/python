# 目的：攝影機 → 用 MediaPipe FaceMesh 偵測臉 → 以「z 最小」近似鼻尖位置
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

            # 7-1)畫出
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style()
            )
         
            # 7-2) 取得「最靠近鏡頭」的點：z 值越小越靠近鏡頭
            #      每個 landmark 都有 (x, y, z)；x,y 為 0~1 正規化，z 為相對深度
            nose_lm = min(face.landmark, key=lambda lm: lm.z)

            # 7-3) 正規化座標 → 換算為像素座標，方便在畫面上標記
            px, py = int(nose_lm.x * w), int(nose_lm.y * h)

            # 7-4) 在估計的鼻尖位置畫一個小圓點（黃色）
            cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)

            # 7-5) 顯示臉編號（idx）以利區分
            label = f"Face {idx} nose"
            cv2.putText(frame, label, (px + 8, py - 8),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            # 7-6)終端機輸出正規化座標 (0~1) 與相對深度 z
            print(f"[Face {idx}] nose_tip_norm: x={nose_lm.x:.3f}, y={nose_lm.y:.3f}, z={nose_lm.z:.3f}")

    # 8) 顯示畫面視窗
    cv2.imshow("FaceMesh - Nose Tip (ESC to quit)", frame)
    # 9) 每 1ms 讀一次鍵盤，若按下 ESC (27) 就離開
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 10) 釋放攝影機並關閉所有視窗
cap.release()
cv2.destroyAllWindows()
