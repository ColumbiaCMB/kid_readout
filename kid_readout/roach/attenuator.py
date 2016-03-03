import requests

class Attenuator(object):
    def __init__(self,attenuation=0, tempip='http://192.168.1.211/'):
        self.ip = tempip
        self.att = self.set_att(attenuation)

    def get_att(self):
        return float(self.get_query("ATT??"))
    
    def set_att(self, newatt):
        if newatt > 62:
            print "Setting attenuation too high.  Max is 62"
        qatt = "SETATT="+str(newatt)
        return float(self.get_query(qatt))
    
    def get_query(self, query):
        query = self.ip + query
        q = requests.get(query)
        return q.content
