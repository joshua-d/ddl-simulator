import socket
from time import perf_counter
 
# file is currently 20 Mb

IP = socket.gethostbyname(socket.gethostname()) # "192.168.12.1" # socket.gethostbyname(socket.gethostname())
PORT = 4455
ADDR = (IP, PORT)
FORMAT = "utf-8"
N_SENDS = 200
 
def main():
    # print('reading file')
    file = open("data.txt", "r")
    data = file.read().encode(FORMAT)

    print(f'data size: {len(data) / 1_000_000} MB')

    print('connecting to %s' % IP)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)
 
    print('sending')
    total_start = perf_counter()
    for i in range(N_SENDS):
        client.send(data)
    total_end = perf_counter()

    print(f'Total time: {total_end - total_start}')
    print(f'Total data sent: {len(data) * N_SENDS / 1_000_000} MB')

    # print('closing file')
    file.close()

    # print('closing client')
    client.close()
 
 
if __name__ == "__main__":
    main()