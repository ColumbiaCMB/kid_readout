import subprocess

# use this as check_output('ssh root@roach "pgrep -f bof$"',shell=True)

def check_output(*popenargs, **kwargs):
    r"""Run command with arguments and return its output as a byte string.

    If the exit code was non-zero it raises a CalledProcessError.  The
    CalledProcessError object will have the return code in the returncode
    attribute and output in the output attribute.

    The arguments are the same as for the Popen constructor.  Example:

    >>> check_output(["ls", "-l", "/dev/null"])
    'crw-rw-rw- 1 root root 1, 3 Oct 18  2007 /dev/null\n'

    The stdout argument is not allowed as it is used internally.
    To capture standard error in the result, use stderr=STDOUT.

    >>> check_output(["/bin/sh", "-c",
    ...               "ls -l non_existent_file ; exit 0"],
    ...              stderr=STDOUT)
    'ls: non_existent_file: No such file or directory\n'
    
    copied from python2.7
    """
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd)
    return output

def start_server(buff_name):
    # Buffname is ppout or ppout0, usually.
    a = check_output('ssh root@roach "pgrep -f bof$"', shell=True)[:4]
    # Slicing is to avoid the newline return.
    print a
    
    # Kill existing processes
    try:
    # Try/except block
        c = check_output('ssh root@roach pkill -f pingpong', shell=True)
    # Problem here: if there is no process to kill, the program ends in an error.
    except subprocess.CalledProcessError:
        pass
        
    
    # remote_command = 'ssh root@roach "/boffiles/udp/channel_pingpong_fileserver %s %s"' % (a,buff_name)
    remote_command = 'nohup ssh root@roach "/boffiles/udp/channel_pingpong_fileserver %s %s" < /dev/null &> /dev/null &' % (a, buff_name)
    print remote_command
    b = check_output(remote_command, shell=True)
    # The goal of the extra stuff is to not wait for a return. This doesn't seem to work.

'''Check output sends commands to the terminal. The goal here is to write a 
program that automatically starts up the roach, initializes channels, etc.'''
