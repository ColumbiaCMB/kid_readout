from kid_readout.utils.time_tools import date_to_unix_time


by_unix_time_table = [
    dict(date='2014-12-11',
         description='STAR Cryo 4x5 130919 0813f12 ASU Al horn package, AR chip, aluminum tape over horns, '
                     'copper shield soldered shut around package, steelcast coax filters, small hole in copper and '
                     'aluminum shields to let in 1550 nm and red LED light',
         optical_state='light',
         chip_id='130919 0813f12',
    ),
    dict(date='2014-12-01',
         description='STAR Cryo 4x5 130919 0813f12 ASU Al horn package, AR chip, aluminum tape over horns, '
                     'copper shield soldered shut around package, steelcast coax filters',
         optical_state='dark',
         chip_id='130919 0813f12',
    ),
    dict(date='2014-08-11',
         description='STAR Cryo 3x3 140423 0813f4 JPL window package, Al tape with hole for red LED',
         optical_state='dark',
         chip_id='140423 0813f4'),
    dict(date='2014-07-28',
         description='STAR Cryo 5x4 130919 0813f12 Aluminum package, taped horns',
         optical_state='dark',
         chip_id='130919 0813f12'),
    dict(date='2014-04-14',
         description='STAR Cryo 3x3 0813f5 JPL window package, Al tape cover, encased in copper tamale',
         optical_state='dark',
         chip_id='130919 0813f5'),
    dict(date='2014-02-21',
         description='STAR Cryo 3x3 0813f5 JPL window package, Al tape cover part 2',
         optical_state='dark',
         chip_id='130919 0813f5'),
    dict(date='2014-02-10',
         description='STAR Cryo 3x3 0813f5 JPL window package, Al tape cover',
         optical_state='dark',
         chip_id='130919 0813f5'),
    ]

by_unix_time_table.sort(key=lambda x: date_to_unix_time(x['date']))
_unix_time_index = [date_to_unix_time(x['date']) for x in by_unix_time_table]
