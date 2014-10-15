import bisect
from kid_readout.utils.time_tools import date_to_unix_time

by_unix_time_table = [
                ('2014-10-10', 'STAR Cryo 4x5 130919 0813f12 ASU Al horn package, AR chip, LPF, copper shield, waveguide with air spacer, absorber tied to 1 K', 'light'),
                ('2014-09-11', 'STAR Cryo 4x5 130919 0813f12 ASU Al horn package, AR chip, LPF, copper shield, waveguide with air spacer', 'light'),
                ('2014-08-26', 'STAR Cryo 4x5 130919 0813f12 ASU Al horn package, AR chip, LPF, copper shield, waveguide with Stycast spacer', 'light'),
                ('2014-08-18', 'STAR Cryo 4x5 130919 0813f12 ASU Al horn package, AR chip, no LPF, copper shield, IR LED fiber', 'light'),
                ('2014-08-12', 'STAR Cryo 4x5 140423 0813f12 Al horn package, AR chip, LPF, copper shield, IR LED fiber', 'light'),
                ('2014-07-30', 'STAR Cryo 4x5 HFdip 0813f9 Al horn package, AR chip, LPF, copper shield, IR LED fiber', 'light'),
                ('2014-07-03', 'STAR Cryo 4x5 0813f12 Al horn package, AR chip, LPF, copper shield, IR LED fiber', 'light'),
                ('2014-04-28', 'STAR Cryo 4x5 0813f12 Al horn package, AR chip, LPF, copper shield', 'light'),
                ('2014-04-16', 'STAR Cryo 4x5 0813f12 Al horn package, AR chip, fully taped', 'dark'),
                ('2014-04-10', 'STAR Cryo 4x5 0813f12 Al horn package, AR chip, Al tape over horns, copper shield', 'dark'),
                ('2014-04-04', 'STAR Cryo 4x5 0813f12 Al horn package, AR chip, LPF, Al tape over horns', 'dark'),
                ('2014-03-28', 'STAR Cryo 4x5 0813f12 Al horn package, AR chip, LPF, Al tape over a few horns', 'light'),
                ('2014-03-19', 'STAR Cryo 4x5 0813f12 Al horn package, AR chip, LPF, broken connection', 'light'),
                ('2014-02-27', 'STAR Cryo 4x5 0813f10 Cu horn package, LPF', 'light'),
                ('2014-01-28', 'STAR Cryo 4x5 0813f10 Cu horn package, no LPF', 'light'),
                ]

by_unix_time_table.sort(key = lambda x: date_to_unix_time(x[0]))
_unix_time_index = [date_to_unix_time(x[0]) for x in by_unix_time_table]

