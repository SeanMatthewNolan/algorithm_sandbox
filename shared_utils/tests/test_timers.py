import sys
import logging

import pytest

from shared_utils.timers import Timer, PerfTimer, ProcessTimer, TimerNS, PerfTimerNS, ProcessTimerNS

"""
Set up logger for testing
"""
LOG_LEVEL = logging.INFO

test_logger = logging.getLogger('test_logger')
test_logger.setLevel(LOG_LEVEL)

stream = logging.StreamHandler(sys.stdout)
stream.setLevel(LOG_LEVEL)
stream_formatter = logging.Formatter('%(funcName)s:%(message)s')
stream.setFormatter(stream_formatter)
test_logger.addHandler(stream)


@pytest.mark.parametrize('timer', [Timer, PerfTimer, ProcessTimer, Timer, TimerNS, PerfTimerNS, ProcessTimerNS])
@pytest.mark.parametrize('logger', [None, test_logger])
def test_timers(timer, logger):
    with timer(logger=logger, logging_level=LOG_LEVEL, log_prefix='test_timer:'):
        _ = sum([i for i in range(1_000_000)])
