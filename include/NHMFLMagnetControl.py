'''
Version = 0.2
Authors = Sasha Zibrov & Carlos Kometter
'''

import socket, struct

class NHMFLMagnetControl(object):
    def __init__(self, address='localhost', port=6341, timeout=5 ):
        self.serv = (address, port)
        self.timeout = timeout
        self.connect()

    def connect(self, iter=3, error=None):
        '''
        Connects to server. If fails, it will try for 'iter' times.
        '''
        for n in xrange(3):
            try:
                print('trying to connect {} th time'.format(n))
                try:
                    client = self.client
                    client.shutdown(1)
                except (NameError, AttributeError):
                    pass
                client = socket.socket();
                client.settimeout(self.timeout)
                failed = client.connect_ex(self.serv)
                if not failed:
                    self.client = client
                    break;
            except Exception as e:
                print(e)

    def status(self):
        '''
        Returns a dictionary with the current status.
        {'Field', 'Setpoint', 'SlewRate', 'Ramp', 'Pause', 'Units'}
        '''
        client = self.client
        try:
            client.send( self.sendData('g'))
            databytes = client.recv(4)  # 4 bytes containing data length
            data = client.recv(self.getByteSize(databytes))
            data = data.split(',')
            mag = {'Field': float(data[0]),
                        'Setpoint': float(data[1]),
                        'SlewRate': float(data[2]),
                        'Ramp': bool(int(data[3])),
                        'Pause': bool(int(data[4])),
                        'Units': bool(int(data[5]))
                       }
            return mag
        except socket.error as e:
            self.connect(error = e)
            self.status()


    def setpoint(self, field):
        '''
        Sets field set point. Returns True
        '''
        client = self.client
        try:
            client.send( self.sendData('s'+str(field)))
            return True
        except socket.error as e:
            self.connect(error = e)
            self.setpoint(field)

    def rate(self, rate):
        '''
        Sets rate at which field increases. Returns True
        '''
        client = self.client
        try:
            client.send( self.sendData('r'+str(rate)))
            return True
        except socket.error as e:
            self.connect(error = e)
            self.rate(field)

    def ramp(self, updown=True):
        '''
        If set to True, ramps field to set point. If set to False, ramps
        field to 0 T or lowest field.
        '''
        client = self.client
        try:
            client.send( self.sendData('u'+str(int(updown))))
            return True
        except socket.error as e:
            self.connect(error = e)
            self.ramp(updown)

    def units(self, un='tesla'):
        '''
        Sets units. 0 = Tesla 1 == kAmps
        '''
        client = self.client
        try:
            if un == 'tesla':
                un = 0
            elif un == 'kAmps':
                un = 1
            client.send( self.sendData('n'+str(int(un))))
            return True
        except socket.error as e:
            self.connect(error = e)
            self.units(un)

    def pause(self, b=True):
        '''
        True: Pause?
        '''
        client = self.client
        try:
            client.send( self.sendData('p'+str(int(b))))
            return True
        except socket.error as e:
            self.connect(error = e)
            self.pause(b)

    def shutdown(self):
        '''
        Shutdowns connection.
        '''
        self.client.shutdown(1)


    @staticmethod
    def sendData(s):
        return struct.pack('I', len(s)) + s

    @staticmethod
    def getByteSize(data):
        return int(struct.unpack('I', data)[0])
