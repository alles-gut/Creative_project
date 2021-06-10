import socket

host = '192.168.0.9'

for port in range(10000,20000):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((host, port))
        s.send(('getBanner\n'))
        banner = s.recv(1024)
        if result == 0:
                print "[+] Port %s tcp/open" % port
                print "[+] Banner: %s" % banner
        s.close()

