import time
import RPi.GPIO as GPIO
import board
import busio
import digitalio
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import adafruit_dht

# --- GPIO Pin Mapping (Physical to BCM Translation) ---
PIN_SERVO = 18       # Physical Pin 12 (Hardware PWM0)
PIN_BUZZER = 23      # Physical Pin 16
PIN_MOTOR_IN1 = 17   # Physical Pin 11
PIN_MOTOR_IN2 = 27   # Physical Pin 13
PIN_MOTOR_ENA = 12   # Physical Pin 32 (Hardware PWM0 alternative)
PIN_DHT11 = board.D4 # Physical Pin 7

# LDR Digital Pin D0
PIN_LDR_DO = 16      # Physical Pin 36 (BCM GPIO 16)

# TCS3200 Color Sensor Pins (Physical Pins 29, 31, 35, 33, 37)
PIN_S0 = 5           # Physical Pin 29
PIN_S1 = 6           # Physical Pin 31
PIN_S2 = 19          # Physical Pin 35
PIN_S3 = 13          # Physical Pin 33
PIN_OUT = 26         # Physical Pin 37

# --- Setup GPIO ---
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

output_pins = [
    PIN_BUZZER,
    PIN_MOTOR_IN1,
    PIN_MOTOR_IN2,
    PIN_MOTOR_ENA,
    PIN_S0,
    PIN_S1,
    PIN_S2,
    PIN_S3,
    PIN_SERVO
]

for pin in output_pins:
    GPIO.setup(pin, GPIO.OUT)

# Inputs
GPIO.setup(PIN_OUT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(PIN_LDR_DO, GPIO.IN)  # LDR Digital threshold input

# Setup PWM Channels
servo_pwm = GPIO.PWM(PIN_SERVO, 50)   # 50Hz for SG90/MG90S Servo
servo_pwm.start(0)

motor_pwm = GPIO.PWM(PIN_MOTOR_ENA, 100) # 100Hz for DC motor speed control
motor_pwm.start(0)

# Set TCS3200 Frequency Scaling to 20%
GPIO.output(PIN_S0, GPIO.HIGH)
GPIO.output(PIN_S1, GPIO.LOW)

# Setup MCP3008 ADC via Hardware SPI (Pins 19, 21, 23, 24)
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = digitalio.DigitalInOut(board.D8)
mcp = MCP.MCP3008(spi, cs)

# Analog Sensor Channels
ldr_channel = AnalogIn(mcp, MCP.P0)    # LDR A0 connected to MCP3008 CH0
mq135_channel = AnalogIn(mcp, MCP.P1)  # MQ-135 connected to MCP3008 CH1

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
    print("    REAL HARDWARE DIAGNOSTICS SUITE      ")
    print("========================================")

    print("\n[1] Testing Active Buzzer (Pin 16 / GPIO 23)...")
    GPIO.output(PIN_BUZZER, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(PIN_BUZZER, GPIO.LOW)
    print(" -> Buzzer: OK")

    print("\n[2] Testing Deflection Servo Arm (Pin 12 / GPIO 18)...")
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

    print("\n[4] Reading LDR Sensor (A0 -> ADC CH0 | D0 -> GPIO 16)...")
    ldr_voltage = ldr_channel.voltage
    ldr_raw = ldr_channel.value
    ldr_digital = GPIO.input(PIN_LDR_DO)
    state_str = "LIGHT DETECTED (LOW)" if ldr_digital == 0 else "DARK / TRIGGER OFF (HIGH)"
    print(f" -> Analog Voltage: {ldr_voltage:.2f} V | Raw ADC: {ldr_raw}")
    print(f" -> Digital Output (D0): {ldr_digital} ({state_str})")

    print("\n[5] Reading MQ-135 Gas Sensor (MCP3008 Channel 1)...")
    mq135_voltage = mq135_channel.voltage
    mq135_raw = mq135_channel.value
    print(f" -> MQ-135 Voltage: {mq135_voltage:.2f} V | Raw ADC Value: {mq135_raw}")

    print("\n[6] Reading TCS3200 Color Sensor Frequencies...")
    r = read_tcs3200_channel(GPIO.LOW, GPIO.LOW)
    b = read_tcs3200_channel(GPIO.LOW, GPIO.HIGH)
    g = read_tcs3200_channel(GPIO.HIGH, GPIO.HIGH)
    print(f" -> Spectral Frequencies: R={r} | G={g} | B={b}")

    print("\n[7] Reading DHT11 Sensor (Pin 7 / GPIO 4)...")
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