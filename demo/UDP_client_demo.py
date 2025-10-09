import socket

HOST = '127.0.0.1'   # ← 這裡改成 127.0.0.1（同機測試），或改成 server 的實際 IP
PORT = 7000
server_addr = (HOST, PORT)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    print('####client####')
    outdata = input('please input message: ')
    print('sendto ' + str(server_addr) + ': ' + outdata)
    s.sendto(outdata.encode(), server_addr)
    
    indata, addr = s.recvfrom(1024)
    print('recvfrom ' + str(addr) + ': ' + indata.decode())

