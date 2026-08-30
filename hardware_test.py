import time
import RPi.GPIO as GPIO
import board
import busio
import digitalio
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import adafruit_dht

# --- GPIO Pin Mapping ---
PIN_SERVO = 18       # SG90/MG90S Servo PWM
PIN_BUZZER = 23      # Active Buzzer
PIN_MOTOR_IN1 = 24   # L298N Conveyor IN1
PIN_MOTOR_IN2 = 25   # L298N Conveyor IN2
PIN_MOTOR_ENA = 12   # L298N Speed PWM
PIN_DHT11 = board.D4 # DHT11 Data Pin

# TCS3200 Color Sensor Pins (Conflict-Free)
PIN_S0 = 17
PIN_S1 = 27
PIN_S2 = 22
PIN_S3 = 5
PIN_OUT = 6

# --- Setup GPIO ---
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in [PIN_BUZZER, PIN_MOTOR_IN1, PIN_MOTOR_IN2, PIN_MOTOR_ENA, PIN_S0, PIN_S1, PIN_S2, PIN_S3]:
    GPIO.setup(pin, GPIO.OUT)
GPIO.setup(PIN_OUT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(PIN_SERVO, GPIO.OUT)

servo_pwm = GPIO.PWM(PIN_SERVO, 50) # 50Hz
servo_pwm.start(0)

motor_pwm = GPIO.PWM(PIN_MOTOR_ENA, 100) # 100Hz
motor_pwm.start(0)

# Set TCS3200 Frequency Scaling to 20%
GPIO.output(PIN_S0, GPIO.HIGH)
GPIO.output(PIN_S1, GPIO.LOW)

# Setup MCP3008 ADC (SPI) for MQ-135
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = digitalio.DigitalInOut(board.D8)
mcp = MCP.MCP3008(spi, cs)
mq135_channel = AnalogIn(mcp, MCP.P0)

# Setup DHT11 Sensor
dht_sensor = adafruit_dht.DHT11(PIN_DHT11)

def set_servo_angle(angle):
    duty = 2.5 + (angle / 18.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    servo_pwm.ChangeDutyCycle(0)

def read_tcs3200_channel(s2_val, s3_val):
    GPIO.output(PIN_S2, s2_val)
    GPIO.output(PIN_S3, s3_val)
    time.sleep(0.02)
    start = time.time()
    count = 0
    while time.time() - start < 0.05:
        if GPIO.input(PIN_OUT) == GPIO.LOW:
            count += 1
    return count

def run_hardware_diagnostics():
    print("========================================")
    print("   REAL HARDWARE DIAGNOSTICS SUITE      ")
    print("========================================")
    
    print("\n[1] Testing Active Buzzer...")
    GPIO.output(PIN_BUZZER, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(PIN_BUZZER, GPIO.LOW)
    print(" -> Buzzer: OK")

    print("\n[2] Testing Deflection Servo Arm...")
    set_servo_angle(90)
    time.sleep(1)
    set_servo_angle(0)
    print(" -> Servo: OK")

    print("\n[3] Testing Conveyor DC Motor (L298N)...")
    GPIO.output(PIN_MOTOR_IN1, GPIO.HIGH)
    GPIO.output(PIN_MOTOR_IN2, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(70)
    time.sleep(2)
    motor_pwm.ChangeDutyCycle(0)
    print(" -> Motor: OK")

    print("\n[4] Reading MQ-135 Gas Sensor via MCP3008 ADC...")
    voltage = mq135_channel.voltage
    raw_adc = mq135_channel.value
    print(f" -> Voltage: {voltage:.2f} V | Raw ADC Value: {raw_adc}")

    print("\n[5] Reading TCS3200 Color Sensor Frequencies...")
    r = read_tcs3200_channel(GPIO.LOW, GPIO.LOW)
    b = read_tcs3200_channel(GPIO.LOW, GPIO.HIGH)
    g = read_tcs3200_channel(GPIO.HIGH, GPIO.HIGH)
    print(f" -> Spectral Frequencies: R={r} | G={g} | B={b}")

    print("\n[6] Reading DHT11 Sensor...")
    try:
        t = dht_sensor.temperature
        h = dht_sensor.humidity
        print(f" -> Temperature: {t} C | Humidity: {h} %")
    except RuntimeError as e:
        print(f" -> DHT11 retry: {e.args[0]}")

if __name__ == "__main__":
    try:
        run_hardware_diagnostics()
    finally:
        servo_pwm.stop()
        motor_pwm.stop()
        GPIO.cleanup()
        print("\nDiagnostics complete. Cleaned up all GPIO pins.")