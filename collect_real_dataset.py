import os
import time
import cv2
import pandas as pd
import RPi.GPIO as GPIO
import board
import busio
import digitalio
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import adafruit_dht

DATASET_DIR = "real_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
CSV_FILE = os.path.join(DATASET_DIR, "real_multimodal_data.csv")

PIN_DHT11 = board.D4
PIN_S0, PIN_S1, PIN_S2, PIN_S3, PIN_OUT = 17, 27, 22, 5, 6

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [PIN_S0, PIN_S1, PIN_S2, PIN_S3]:
    GPIO.setup(pin, GPIO.OUT)
GPIO.setup(PIN_OUT, GPIO.IN, pull_up_down=GPIO.PUD_UP)

GPIO.output(PIN_S0, GPIO.HIGH)
GPIO.output(PIN_S1, GPIO.LOW)

spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = digitalio.DigitalInOut(board.D8)
mcp = MCP.MCP3008(spi, cs)
mq135_channel = AnalogIn(mcp, MCP.P0)
dht_sensor = adafruit_dht.DHT11(PIN_DHT11)

def read_color():
    GPIO.output(PIN_S2, GPIO.LOW)
    GPIO.output(PIN_S3, GPIO.LOW)
    time.sleep(0.02)
    r = sum(1 for _ in range(100) if GPIO.input(PIN_OUT) == GPIO.LOW)

    GPIO.output(PIN_S2, GPIO.HIGH)
    GPIO.output(PIN_S3, GPIO.HIGH)
    time.sleep(0.02)
    g = sum(1 for _ in range(100) if GPIO.input(PIN_OUT) == GPIO.LOW)

    GPIO.output(PIN_S2, GPIO.LOW)
    GPIO.output(PIN_S3, GPIO.HIGH)
    time.sleep(0.02)
    b = sum(1 for _ in range(100) if GPIO.input(PIN_OUT) == GPIO.LOW)
    return r, g, b

def capture_real_sample(class_name, label_id, sample_id):
    cam = cv2.VideoCapture(0)
    for _ in range(5):
        cam.read()
    ret, frame = cam.read()
    cam.release()

    if not ret:
        print("[ERROR] Camera frame capture failed!")
        return

    # 1. Save Physical Image Frame
    class_folder = os.path.join(DATASET_DIR, class_name)
    os.makedirs(class_folder, exist_ok=True)
    img_filename = os.path.join(class_folder, f"{class_name}_{sample_id}.jpg")
    cv2.imwrite(img_filename, frame)

    # 2. Read Physical Sensors
    try:
        temp = dht_sensor.temperature
        hum = dht_sensor.humidity
    except:
        temp, hum = 25.0, 60.0

    gas_voltage = mq135_channel.voltage
    gas_raw = mq135_channel.value
    r, g, b = read_color()

    # 3. Save to Real Multimodal CSV
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_id": sample_id,
        "class_name": class_name,
        "label": label_id,
        "image_path": img_filename,
        "temp_c": temp,
        "humidity_pct": hum,
        "gas_voltage": round(gas_voltage, 3),
        "gas_raw": gas_raw,
        "color_r": r,
        "color_g": g,
        "color_b": b
    }

    df = pd.DataFrame([row])
    if not os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, index=False)
    else:
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)

    print(f"[LOGGED] {class_name} (#{sample_id}) | Temp: {temp}C | Hum: {hum}% | Gas: {gas_voltage:.2f}V | RGB: ({r},{g},{b})")

if __name__ == "__main__":
    CLASSES = [
        "FreshApple", "FreshBanana", "FreshPotato", "FreshTomato",
        "RottenApple", "RottenBanana", "RottenPotato", "RottenTomato"
    ]
    print("--- REAL MULTIMODAL PRODUCE LOGGER ---")
    for idx, c in enumerate(CLASSES):
        print(f" [{idx}] {c}")
    
    try:
        count = 1
        while True:
            choice = input("\nEnter Class ID (0-7) to capture sample or 'q' to quit: ")
            if choice.lower() == 'q':
                break
            if choice.isdigit() and int(choice) in range(8):
                capture_real_sample(CLASSES[int(choice)], int(choice), count)
                count += 1
            else:
                print("Invalid selection. Enter a number between 0 and 7.")
    finally:
        GPIO.cleanup()