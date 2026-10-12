import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def process_exercise(exercise, landmarks):
    try:
        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        wrist = [landmarks[15].x, landmarks[15].y]
        hip = [landmarks[23].x, landmarks[23].y]
        knee = [landmarks[25].x, landmarks[25].y]
        ankle = [landmarks[27].x, landmarks[27].y]
    except:
        return False, 0, "جاري تحديد الجسم..."

    is_correct = False
    feedback = ""
    angle = 0

    if exercise == "Squat":
        angle = calculate_angle(hip, knee, ankle)
        if angle < 100:
            is_correct = True
            feedback = "Good Depth / نزول ممتاز"
        else:
            feedback = "Lower Your Hips / انزل أكثر"

    elif exercise == "Push-up":
        angle = calculate_angle(shoulder, elbow, wrist)
        if angle < 90:
            is_correct = True
            feedback = "Great Form / تمرين صحيح"
        else:
            feedback = "Go Deeper / انزل للأسفل"

    elif exercise == "Plank":
        angle = calculate_angle(shoulder, hip, ankle)
        if 160 <= angle <= 180:
            is_correct = True
            feedback = "Perfect Hold / ثبات ممتاز"
        else:
            feedback = "Keep Body Straight / حافظ على استقامة ظهرك"

    return is_correct, int(angle), feedback