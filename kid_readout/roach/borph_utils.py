import subprocess

# use this as check_output('ssh root@roach "pgrep -f bof$"',shell=True)

def _check_output(*popenargs, **kwargs):
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

try:
    from subprocess import check_output
except ImportError:
    check_output = _check_output    
def get_bof_pid(roachip = 'roach'):
    return int(check_output(('ssh root@%s "pgrep -f bof$"' % roachip), shell=True))

def start_server(bof_pid,roachip='roach'):
    try:
        c = check_output(('ssh root@%s pkill -f kid_ppc' % roachip), shell=True)
        print 'process killed'
    except subprocess.CalledProcessError:
        pass
        
    
    # remote_command = 'ssh root@roach "/boffiles/udp/channel_pingpong_fileserver %s %s"' % (a,buff_name)
    #remote_command = 'nohup ssh root@roach "/boffiles/udp/channel_pingpong_fileserver %s %s" < /dev/null &> /dev/null &' % (a, buff_name)
    remote_command = 'ssh root@%s "nohup /boffiles/udp/kid_ppc %s < /dev/null &> /dev/null &"' % (roachip,bof_pid)
    print remote_command
    b = check_output(remote_command, shell=True)
