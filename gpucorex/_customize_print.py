from termcolor import colored

def error(info):
    print(colored('[FAIL]', 'red', attrs=['bold']) + ' ' + info)
    exit(-1)

def warn(info, silence=False):
    if silence: return
    print(colored('[WARN]', 'yellow', attrs=['bold']) + ' ' + info)

def info(info, silence=False):
    if silence: return
    print(colored('[INFO]', 'blue', attrs=['bold']) + ' ' + info)
    
def success(info, silence=False):
    if silence: return
    print(colored('[SUCC]', 'green', attrs=['bold']) + ' ' + info)