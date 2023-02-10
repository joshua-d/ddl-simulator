import socket
from time import perf_counter, sleep
 
IP = '192.168.12.1' # socket.gethostbyname(socket.gethostname())
PORT = 4455
ADDR = (IP, PORT)
SIZE = 5_000_000
FORMAT = "utf-8"
 
def main():
    print("[STARTING] Server is starting.")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    print("[LISTENING] Server is listening. IP: %s" % IP)
 
    while True:
        conn, addr = server.accept()
        print(f"[NEW CONNECTION] {addr} connected.")

        while True:
            data = conn.recv(4096).decode(FORMAT)

 
if __name__ == "__main__":
    main()