import socket
from time import perf_counter, sleep
 
IP = '10.42.0.47' # socket.gethostbyname(socket.gethostname())
PORT = 4455
ADDR = (IP, PORT)
FORMAT = "ascii"

DATA_SIZE = 1_000_000
 
def main():
    print("[STARTING] Server is starting.")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    print("[LISTENING] Server is listening. IP: %s" % IP)
 
    while True:
        conn, addr = server.accept()
        print(f"[NEW CONNECTION] {addr} connected.")

        bytes_received = 0
        start_time = perf_counter()
        while True:
            data = conn.recv(4096)
            bytes_received += len(data)
            if bytes_received == DATA_SIZE:
                end_time = perf_counter()
                break

        print('received all')
        print(f'Total time: {end_time - start_time}')
        print(f'Bitrate: {DATA_SIZE * 8 / (end_time - start_time) / 1_000_000} Mbps')


 
if __name__ == "__main__":
    main()