import socket
from time import perf_counter
 
IP = socket.gethostbyname(socket.gethostname())
PORT = 4455
ADDR = (IP, PORT)
SIZE = 2_500_000
FORMAT = "utf-8"
N_SENDS = 200
 
def main():
    print("[STARTING] Server is starting.")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    print("[LISTENING] Server is listening. IP: %s" % IP)
 
    while True:
        conn, addr = server.accept()
        print(f"[NEW CONNECTION] {addr} connected.")
 
        total_start = perf_counter()
        for i in range(N_SENDS):
            data = conn.recv(SIZE).decode(FORMAT)
        total_end = perf_counter()
        
        print(f'Total time: {total_end - total_start}')
 
if __name__ == "__main__":
    main()