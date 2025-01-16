# sender.py
import serial
import time
import json

def send_array(serial_port, baud_rate, array):
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            # Chuyển mảng thành chuỗi JSON
            data = json.dumps(array)
            # Mã hóa thành bytes và gửi
            ser.write(data.encode('utf-8'))
            print(f"Gửi dữ liệu: {data}")
    except serial.SerialException as e:
        print(f"Lỗi khi mở cổng serial: {e}")

if __name__ == "__main__":
    # Cấu hình cổng serial (thay thế bằng cổng của bạn, ví dụ: COM3 trên Windows hoặc /dev/ttyUSB0 trên Linux)
    SERIAL_PORT = '/dev/pts/3'  # Hoặc '/dev/ttyUSB0'
    BAUD_RATE = 9600

    # Mảng số cần gửi
    array_to_send = [1, 2, 3, 4, 5]

    while True:
        send_array(SERIAL_PORT, BAUD_RATE, array_to_send)
        time.sleep(2)  # Gửi mỗi 2 giây