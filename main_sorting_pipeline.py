import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO

CLASSES = [
    'FreshApple', 'FreshBanana', 'FreshPotato', 'FreshTomato',
    'RottenApple', 'RottenBanana', 'RottenPotato', 'RottenTomato'
]

PIN_SERVO = 18       # SG90 Reject Gate PWM
PIN_BUZZER = 23      # Active Alarm Buzzer
PIN_MOTOR_IN1 = 24   # L298N Conveyor IN1
PIN_MOTOR_IN2 = 25   # L298N Conveyor IN2
PIN_MOTOR_ENA = 12   # L298N PWM Speed

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [PIN_BUZZER, PIN_MOTOR_IN1, PIN_MOTOR_IN2, PIN_MOTOR_ENA, PIN_SERVO]:
    GPIO.setup(pin, GPIO.OUT)

servo_pwm = GPIO.PWM(PIN_SERVO, 50)
servo_pwm.start(0)
motor_pwm = GPIO.PWM(PIN_MOTOR_ENA, 100)
motor_pwm.start(0)

# Load Edge TFLite Model
interpreter = tflite.Interpreter(model_path="models/vegetable_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def start_conveyor(speed=60):
    GPIO.output(PIN_MOTOR_IN1, GPIO.HIGH)
    GPIO.output(PIN_MOTOR_IN2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(speed)

def stop_conveyor():
    motor_pwm.ChangeDutyCycle(0)

def deflect_rotten():
    print(">>> DEFECT DETECTED: Actuating Rejection Mechanism <<<")
    GPIO.output(PIN_BUZZER, GPIO.HIGH)
    servo_pwm.ChangeDutyCycle(7.5) # 90 degrees
    time.sleep(1.2)
    servo_pwm.ChangeDutyCycle(2.5) # Return to 0 degrees
    GPIO.output(PIN_BUZZER, GPIO.LOW)
    time.sleep(0.3)
    servo_pwm.ChangeDutyCycle(0)

def run_live_pipeline():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cam.isOpened():
        print("[ERROR] Camera not accessible.")
        return

    print("==================================================")
    print("   REAL-TIME VEGETABLE SORTING SYSTEM RUNNING     ")
    print("==================================================")
    start_conveyor(speed=60)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img / 127.5) - 1.0
            input_tensor = np.expand_dims(img, axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            pred_idx = np.argmax(preds)
            confidence = preds[pred_idx] * 100.0
            pred_class = CLASSES[pred_idx]

            if confidence > 80.0:
                print(f"[DETECTED] {pred_class} ({confidence:.1f}%)")
                if "Rotten" in pred_class:
                    stop_conveyor()
                    deflect_rotten()
                    start_conveyor(speed=60)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping conveyor system...")
    finally:
        stop_conveyor()
        servo_pwm.stop()
        motor_pwm.stop()
        GPIO.cleanup()
        cam.release()

if __name__ == "__main__":
    run_live_pipeline()