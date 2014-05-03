from kid_readout.utils.time_tools import date_to_unix_time

by_unix_time_table = [('2014-04-14', 'STAR Cryo 3x3 0813f5 JPL window package, Al tape cover, encased in copper tamale', 'dark'),
                      ('2014-02-21', 'STAR Cryo 3x3 0813f5 JPL window package, Al tape cover part 2', 'dark'),
                      ('2014-02-10', 'STAR Cryo 3x3 0813f5 JPL window package, Al tape cover', 'dark'),
                ]

by_unix_time_table.sort(key = lambda x: date_to_unix_time(x[0]))
_unix_time_index = [date_to_unix_time(x[0]) for x in by_unix_time_table]

