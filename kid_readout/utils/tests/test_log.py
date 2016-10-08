from kid_readout.utils import log

def test_log_file_path():
    log.file_handler()
    log.file_handler(__file__)