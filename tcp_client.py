import socket
from time import perf_counter
 
# file is currently 20 Mb

IP = '192.168.12.1'
PORT = 4455
ADDR = (IP, PORT)
FORMAT = "utf-8"
N_SENDS = 5
 
def main():
    # print('reading file')
    file = open("data3.txt", "r")
    data = file.read().encode(FORMAT)

    print(f'data size: {len(data) / 1_000_000} MB, {len(data)} B')

    print('connecting to %s' % IP)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)
 
    print('sending')
    total_start = perf_counter()
    for i in range(N_SENDS):
        client.send(data)
        print(f'sent {i}')
    total_end = perf_counter()

    print(f'Total time: {total_end - total_start}')
    print(f'Total data sent: {len(data) * N_SENDS / 1_000_000} MB')
    print(f'Bitrate: {len(data) * N_SENDS * 8 / (total_end - total_start) / 1_000_000} Mbps')

    # print('closing file')
    file.close()

    # print('closing client')
    client.close()
 
 
if __name__ == "__main__":
    main()