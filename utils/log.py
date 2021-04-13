
class Logger():

    def __init__(self, path):
        self.log_file = open(path+"log.txt", "w")

    def __del__(self):
        self.log_file.close()

    def log(self, log_str):
        log_str = str(log_str)
        self.log_file.write(log_str+"\n")
        self.log_file.flush()
        print(log_str)



