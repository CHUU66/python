import socket
import cv2                      
import mediapipe as mp

#== UDP設置 ==
HOST = '127.0.0.1'
PORT = 7000

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#== MediaPipe Hands ==
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)

hands = mp.solutions.hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles


while True:
    ok, frame = cap.read()     
    if not ok:
        break                  

    frame = cv2.flip(frame,1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 偵測手部
    result = hands.process(rgb)
    
    if result.multi_hand_landmarks:
        
        # 取得第一隻手
        hand_landmarks = result.multi_hand_landmarks[0]
        # 取出食指尖 
        tip = hand_landmarks.landmark[8]
        
        msg=f"{tip.x:.3f},{tip.y:.3f}"
        px, py = int(tip.x * w), int(tip.y * h)
        cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)
        
        s.sendto(msg.encode("utf-8"),(HOST,PORT))
        
    cv2.imshow("Hand Tracking (ESC to quit)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
s.close()