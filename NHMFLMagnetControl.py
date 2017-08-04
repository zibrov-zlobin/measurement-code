import socket, struct

class NHMFLMagnetControl(object):
    def __init__(self, address='localhost', port=6341, timeout=2 ):
        self.serv = (address, port)
        self.timeout = timeout

    def status(self):
        client = socket.socket();
        client.settimeout(self.timeout)
        client.connect(self.serv)
        client.send( self.sendData('g'))
        databytes = client.recv(4)  # 4 bytes containing data length
        data = client.recv(self.getByteSize(databytes))
        client.close()
        data = data.split(',')
        mag = {'Field': float(data[0]),
                    'Setpoint': float(data[1]),
                    'SlewRate': float(data[2]),
                    'Ramp': bool(data[3]),
                    'Pause': bool(data[3]),
                    'Units': bool(data[4])
                   }
        return mag


    def setpoint(self, field):
        client = socket.socket();
        client.settimeout(self.timeout)
        client.connect(self.serv)
        client.send( self.sendData('s'+str(field)))
        client.close()
        return True

    def rate(self, rate):
        client = socket.socket();
        client.settimeout(self.timeout)
        client.connect(self.serv)
        client.send( self.sendData('r'+str(rate)))
        client.close()
        return True

    def ramp(self, updown=True):
        client = socket.socket();
        client.settimeout(self.timeout)
        client.connect(self.serv)
        client.send( self.sendData('u'+str(int(updown))))
        client.close()
        return True

    def units(self, un='tesla'):
        if un == 'tesla':
            un = 0
        elif un == 'kAmps':
            un = 1
        client = socket.socket();
        client.settimeout(self.timeout)
        client.connect(self.serv)
        client.send( self.sendData('n'+str(int(un))))
        client.close()
        return True

    def pause(self, b=True):
        client = socket.socket();
        client.settimeout(self.timeout)
        client.connect(self.serv)
        client.send( self.sendData('p'+str(int(b))))
        client.close()
        return True


    @staticmethod
    def sendData(s):
        return struct.pack('I', len(s)) + s

    @staticmethod
    def getByteSize(data):
        return int(struct.unpack('I', data)[0])
