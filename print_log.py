from datetime import datetime
import os
import time

SET_DEBUG = True

CONST_LOG_LEVELS = { 'verbose': 0, 'v': 0, 'info': 1, 'i': 1, 'warning': 2, 'w': 2, 'error': 3, 'e': 3 }
CONST_LOG_LEVELS_ABBR = { 0: 'v', 1: 'i', 2: 'w', 3: 'e' }
CONST_LOG_LEVEL_LIMIT = 'v'

LOG_BUFFER = []
SET_BUFFER_FLUSH_COUNT = 10

SET_PRINTLOG_OUTPUT_PATH = ""
SET_LOG_FILE = ""
SET_JOB_ID = ""

def set_log_path(path, job_id):
    global SET_PRINTLOG_OUTPUT_PATH
    global SET_LOG_FILE
    global SET_JOB_ID

    if os.path.exists(path) == False:
        os.makedirs(path)

    SET_PRINTLOG_OUTPUT_PATH = path
    SET_JOB_ID = job_id

    now = time.localtime()
    log_file_name = "%s_%04d%02d%02d_%02d%02d%02d.log" % (str(job_id), now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    SET_LOG_FILE = os.path.join(SET_PRINTLOG_OUTPUT_PATH, log_file_name)

    return log_file_name, job_id

def set_flush_counter(log_count:int = 10):
    global SET_BUFFER_FLUSH_COUNT

    if log_count < 0:
        log_count = 1

    SET_BUFFER_FLUSH_COUNT = log_count

def set_debug(flag = False):
    global SET_DEBUG
    SET_DEBUG = flag

def is_debug():
    global SET_DEBUG
    return SET_DEBUG

def getJobId():
    return SET_JOB_ID
    

def print_log(level:str, msg, tag:str, on_screen_display:bool = False, force_flush:bool = False):
    '''
    print_log

    Arguments:
    ------------
    - level = [ 'verbose' | 'v' | 'info' | 'i' | 'warning' | 'w' | 'error' | 'e' ]
    - msg = [ any ], message about whatever you want to logging
    - tag = [ string ], name of log owner (i.e., event occured module)
    - on_screen_display = [ True | False ], if set True then log message(msg) will shows on terminal screen
    - force_flush = [ True | False ], if set True then flush log to file immediately
    '''
    global SET_LOG_FILE
    global SET_DEBUG
    global SET_BUFFER_FLUSH_COUNT
    global LOG_BUFFER

    if level not in CONST_LOG_LEVELS: return
    if CONST_LOG_LEVELS[CONST_LOG_LEVEL_LIMIT] > CONST_LOG_LEVELS[level]: return

    if len(msg) == 0: return

    if SET_PRINTLOG_OUTPUT_PATH == "" or SET_LOG_FILE == "":
        print("ERROR! call set_log_path fisrt")
        return

    log_msg = "{0} {1}/[{2}]: {3}".format(datetime.now(), str.upper(CONST_LOG_LEVELS_ABBR[CONST_LOG_LEVELS[level]]), tag, msg)
    LOG_BUFFER.append(log_msg)

    if on_screen_display == True:
        print(log_msg)

    if len(LOG_BUFFER) > SET_BUFFER_FLUSH_COUNT or force_flush:
        with open(SET_LOG_FILE, 'a', encoding='utf-8') as log_file:
            log_file.write("\n".join(LOG_BUFFER) + "\n")
        LOG_BUFFER.clear()

def print_logv(msg, tag, on_screen_display = False, force_flush = False):
    '''
    print_log for verbose messages

    Arguments:
    ------------
    - msg = [ any ], message about whatever you want to logging
    - tag = [ string ], name of log owner (i.e., event occured module)
    - on_screen_display = [ True | False ], if set True then log message(msg) will shows on terminal screen
    - force_flush = [ True | False ], if set True then flush log to file immediately
    '''
    print_log(level = 'v', msg = msg, tag = tag, on_screen_display = on_screen_display, force_flush = force_flush)

def print_logi(msg, tag, on_screen_display = False, force_flush = False):
    '''
    print_log for informational messages

    Arguments:
    ------------
    - msg = [ any ], message about whatever you want to logging
    - tag = [ string ], name of log owner (i.e., event occured module)
    - on_screen_display = [ True | False ], if set True then log message(msg) will shows on terminal screen
    - force_flush = [ True | False ], if set True then flush log to file immediately
    '''
    print_log(level = 'i', msg = msg, tag = tag, on_screen_display = on_screen_display, force_flush = force_flush)

def print_logw(msg, tag, on_screen_display = False, force_flush = False):
    '''
    print_log for warning messages

    Arguments:
    ------------
    - msg = [ any ], message about whatever you want to logging
    - tag = [ string ], name of log owner (i.e., event occured module)
    - on_screen_display = [ True | False ], if set True then log message(msg) will shows on terminal screen
    - force_flush = [ True | False ], if set True then flush log to file immediately
    '''
    print_log(level = 'w', msg = msg, tag = tag, on_screen_display = on_screen_display, force_flush = force_flush)

def print_loge(msg, tag, on_screen_display = False, force_flush = False):
    '''
    print_log for error messages

    Arguments:
    ------------
    - msg = [ any ], message about whatever you want to logging
    - tag = [ string ], name of log owner (i.e., event occured module)
    - on_screen_display = [ True | False ], if set True then log message(msg) will shows on terminal screen
    - force_flush = [ True | False ], if set True then flush log to file immediately
    '''
    print_log(level = 'e', msg = msg, tag = tag, on_screen_display = on_screen_display, force_flush = force_flush)
