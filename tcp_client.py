import socket
from time import perf_counter
import random
 
# file is currently 20 Mb

IP = '10.42.0.47'
PORT = 4455
ADDR = (IP, PORT)
FORMAT = "ascii"

DATA_SIZE = 1_000_000

 
def main():
    print('generating data')
    data = random.randbytes(DATA_SIZE)

    print(f'data size: {len(data) / 1_000_000} MB, {len(data)} B')

    print('connecting to %s' % IP)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)
 
    print('sending')
    total_start = perf_counter()
    client.sendall(data)
    print('sent all')
    total_end = perf_counter()

    print(f'Total time: {total_end - total_start}')
    print(f'Total data sent: {len(data) / 1_000_000} MB')
    print(f'Bitrate: {len(data) * 8 / (total_end - total_start) / 1_000_000} Mbps')

    # print('closing client')
    client.close()
 
 
if __name__ == "__main__":
    main()