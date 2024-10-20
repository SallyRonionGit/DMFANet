import sys
import time
class Logger(object):
    def __init__(self, outfile):
        # get the output stream
        self.terminal = sys.stdout
        # outfile : the path of the Logger
        self.log_path = outfile
        # get current time and change it's type into string and will print it when initial
        # .write can only be used to write string type
        now = time.strftime("%c")
        # .write(defined in def write(self, message))
        self.write('================ (%s) ================\n' % now)

    # write file(won't overwrite because mode == 'a')
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, mode='a') as f:
            f.write(message)

    # write string, value
    # here message transfer k,v dict into string or .write %.7f will cause error
    def write_dict(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v)
        self.write(message)

    # write string,string
    def write_dict_str(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %s ' % (k, v)
        self.write(message)
    
    # Clear the cache of output (stdout)
    def flush(self):
        self.terminal.flush()

class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        # if start message comes print it and record start time
        # time.time() return second,time.ctime transfer seconds to string("Wed Jun 30 21:49:08 1993")
        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    # the class can be used as(with Timer as my_object:)(Context Management Protocol)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    # progress : the percentage number of the finished work occupy in total
    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        # predict the total time
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        # predict the final finish time
        self.est_finish = int(self.start + self.est_total)

    # time.ctime() already string,str() could be ignored
    # finish time(string)
    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))

    # remain time(string)
    def str_estimated_remaining(self):
        return str(self.est_remaining/3600) + 'h'

    # remain time(int)
    def estimated_remaining(self):
        return self.est_remaining/3600

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out

