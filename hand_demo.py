# 目的：攝影機 → 用 MediaPipe 偵測手部 → 把 21 個關節點畫到畫面上
# 離開鍵：按鍵盤 ESC

import cv2                      # OpenCV：處理攝影機與畫面顯示
import mediapipe as mp          

# 1) 打開預設攝影機（0 = 內建鏡頭；如果外接另一顆，可能要用 1 或 2）
cap = cv2.VideoCapture(0)

# 2) 建立 MediaPipe Hands（偵測器）
#    static_image_mode= *T/F* ：視訊串流模式（False:會持續追蹤，速度快)(True:則每張圖像都會獨立處理)
#    max_num_hands= *number*：最多手部偵測數量
#    min_detection_confidence= *number0~1* : 檢測的最低可信度門檻>用在「第一次找到手/臉」的時候。（0~1，越高越嚴格）
#    min_tracking_confidence= *number0~1* : 追蹤信心值門檻>用在「已經偵測到，接下來持續追蹤」的時候
hands = mp.solutions.hands.Hands(
    static_image_mode=False, 
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3) 導入繪圖實用工具函數
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

while True:
    ok, frame = cap.read()      # 4) 讀取攝影機畫面
    if not ok:
        break                   # 抓不到就離開（例如鏡頭被占用）

    # 5) MediaPipe 要吃 RGB，所以先把 BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 6) 偵測手部
    result = hands.process(rgb)

    # 7) 繪製手部關鍵點和連線
    if result.multi_hand_landmarks:  # 如果有偵測到一隻或多隻手
        h, w, _ = frame.shape # 取得影像尺寸(height,width,channels(如 RGB 通道))
        for hand_landmarks in result.multi_hand_landmarks:
            # 7-1) 畫出 21 個關節與連線（畫在原本的 BGR 影像 frame 上）
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(), #默認樣式
                mp_style.get_default_hand_connections_style()
            )

            # 7-2) 例子：取得食指尖端（id=8）的「正規化座標」(0~1)
            tip = hand_landmarks.landmark[8]
            # 轉為像素座標（整數），方便畫圓點
            px, py = int(tip.x * w), int(tip.y * h)

            # 7-3) 在食指尖端畫一個綠色小圓點  (8半徑)
            cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)

            # 7-4) 同時在終端機印出正規化座標（之後要用這對 x,y 傳給 Unity）
            print(f"index_tip: x={tip.x:.3f}, y={tip.y:.3f}")
            
            #練習區:(得到大拇指)大拇指生成白色圓形
            test=hand_landmarks.landmark[4]
            ox,oy=int(test.x * w),int(test.y * h)
            cv2.circle(frame, (ox, oy), 10, (0, 0, 0), -1)
            

    # 8) 顯示畫面視窗
    cv2.imshow("Hand Tracking (ESC to quit)", frame)

    # 9) 每 1ms 讀一次鍵盤，若按下 ESC (27) 就離開
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 10) 釋放攝影機並關閉所有視窗
cap.release()
cv2.destroyAllWindows()
