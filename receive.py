# receiver.py
import serial
import json

def receive_array(serial_port, baud_rate):
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            print("Đang lắng nghe dữ liệu...")
            while True:
                if ser.in_waiting:
                    # Đọc dữ liệu đến
                    data = ser.read(ser.in_waiting).decode('utf-8').strip()
                    if data:
                        try:
                            # Giải mã JSON thành mảng
                            array = json.loads(data)
                            print(f"Nhận được mảng: {array}")
                        except json.JSONDecodeError:
                            print(f"Dữ liệu không hợp lệ: {data}")
    except serial.SerialException as e:
        print(f"Lỗi khi mở cổng serial: {e}")

if __name__ == "__main__":
    # Cấu hình cổng serial (thay thế bằng cổng của bạn, ví dụ: COM4 trên Windows hoặc /dev/ttyUSB1 trên Linux)
    SERIAL_PORT = '/dev/pts/4'  # Hoặc '/dev/ttyUSB1'
    BAUD_RATE = 9600

    receive_array(SERIAL_PORT, BAUD_RATE)