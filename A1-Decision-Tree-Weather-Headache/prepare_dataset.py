import pandas as pd
import datetime

class DatasetPrep:
    avg_weather_data = None
    work_column = None
    workout_column = None
    headache_column = None

    def read_and_prepare_weather_dataset(self):
        # read month-wise climate data containing hourly weather details from 5 different dataset sources
        # sep climate dataset
        HALIFAX_DOCKYARD_Sep = pd.read_csv('dataset/HALIFAX_DOCKYARD/en_climate_hourly_NS_8202240_09-2019_P1H.csv')
        HALIFAX_KOOTENAY_Sep = pd.read_csv('dataset/HALIFAX_KOOTENAY/en_climate_hourly_NS_8202252_09-2019_P1H.csv')
        HALIFAX_STANFIELD_INTL_A1_Sep = pd.read_csv(
            'dataset/HALIFAX_STANFIELD_INTL_A1/en_climate_hourly_NS_8202251_09-2019_P1H.csv')
        HALIFAX_STANFIELD_INTL_A2_Sep = pd.read_csv(
            'dataset/HALIFAX_STANFIELD_INTL_A2/en_climate_hourly_NS_8202249_09-2019_P1H.csv')
        HALIFAX_WINDSOR_PARK_Sep = pd.read_csv(
            'dataset/HALIFAX_WINDSOR_PARK/en_climate_hourly_NS_8202255_09-2019_P1H.csv')

        # oct climate dataset
        HALIFAX_DOCKYARD_Oct = pd.read_csv('dataset/HALIFAX_DOCKYARD/en_climate_hourly_NS_8202240_10-2019_P1H.csv')
        HALIFAX_KOOTENAY_Oct = pd.read_csv('dataset/HALIFAX_KOOTENAY/en_climate_hourly_NS_8202252_10-2019_P1H.csv')
        HALIFAX_STANFIELD_INTL_A1_Oct = pd.read_csv(
            'dataset/HALIFAX_STANFIELD_INTL_A1/en_climate_hourly_NS_8202251_10-2019_P1H.csv')
        HALIFAX_STANFIELD_INTL_A2_Oct = pd.read_csv(
            'dataset/HALIFAX_STANFIELD_INTL_A2/en_climate_hourly_NS_8202249_10-2019_P1H.csv')
        HALIFAX_WINDSOR_PARK_Oct = pd.read_csv(
            'dataset/HALIFAX_WINDSOR_PARK/en_climate_hourly_NS_8202255_10-2019_P1H.csv')

        # nov climate dataset
        HALIFAX_DOCKYARD_Nov = pd.read_csv('dataset/HALIFAX_DOCKYARD/en_climate_hourly_NS_8202240_11-2019_P1H.csv')
        HALIFAX_KOOTENAY_Nov = pd.read_csv('dataset/HALIFAX_KOOTENAY/en_climate_hourly_NS_8202252_11-2019_P1H.csv')
        HALIFAX_STANFIELD_INTL_A1_Nov = pd.read_csv(
            'dataset/HALIFAX_STANFIELD_INTL_A1/en_climate_hourly_NS_8202251_11-2019_P1H.csv')
        HALIFAX_STANFIELD_INTL_A2_Nov = pd.read_csv(
            'dataset/HALIFAX_STANFIELD_INTL_A2/en_climate_hourly_NS_8202249_11-2019_P1H.csv')
        HALIFAX_WINDSOR_PARK_Nov = pd.read_csv(
            'dataset/HALIFAX_WINDSOR_PARK/en_climate_hourly_NS_8202255_11-2019_P1H.csv')

        # dec climate dataset
        HALIFAX_DOCKYARD_Dec = pd.read_csv('dataset/HALIFAX_DOCKYARD/en_climate_hourly_NS_8202240_12-2019_P1H.csv')
        HALIFAX_KOOTENAY_Dec = pd.read_csv('dataset/HALIFAX_KOOTENAY/en_climate_hourly_NS_8202252_12-2019_P1H.csv')
        HALIFAX_STANFIELD_INTL_A1_Dec = pd.read_csv(
            'dataset/HALIFAX_STANFIELD_INTL_A1/en_climate_hourly_NS_8202251_12-2019_P1H.csv')
        HALIFAX_STANFIELD_INTL_A2_Dec = pd.read_csv(
            'dataset/HALIFAX_STANFIELD_INTL_A2/en_climate_hourly_NS_8202249_12-2019_P1H.csv')
        HALIFAX_WINDSOR_PARK_Dec = pd.read_csv(
            'dataset/HALIFAX_WINDSOR_PARK/en_climate_hourly_NS_8202255_12-2019_P1H.csv')

        # read date/time column and concatinate them
        date_time_col_sep = HALIFAX_DOCKYARD_Sep['Date/Time']
        date_time_col_oct = HALIFAX_DOCKYARD_Oct['Date/Time']
        date_time_col_nov = HALIFAX_DOCKYARD_Nov['Date/Time']
        date_time_col_dec = HALIFAX_DOCKYARD_Dec['Date/Time']
        date_time_col = pd.concat([date_time_col_sep, date_time_col_oct, date_time_col_nov, date_time_col_dec])

        # read following columns month-wise:
        # Date/Time, Temp, Dew Point temp, Rel Hum , Wind Spd, Visibility, Stn Press
        # sep
        halifax_dockyard_sep = HALIFAX_DOCKYARD_Sep[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_kootenay_sep = HALIFAX_KOOTENAY_Sep[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_stanfield_intl_a1_sep = HALIFAX_STANFIELD_INTL_A1_Sep[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_stanfield_intl_a2_sep = HALIFAX_STANFIELD_INTL_A2_Sep[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_windsor_sep = HALIFAX_WINDSOR_PARK_Sep[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]

        # oct
        halifax_dockyard_oct = HALIFAX_DOCKYARD_Oct[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_kootenay_oct = HALIFAX_KOOTENAY_Oct[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_stanfield_intl_a1_oct = HALIFAX_STANFIELD_INTL_A1_Oct[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_stanfield_intl_a2_oct = HALIFAX_STANFIELD_INTL_A2_Oct[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_windsor_oct = HALIFAX_WINDSOR_PARK_Oct[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]

        # nov
        halifax_dockyard_nov = HALIFAX_DOCKYARD_Nov[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_kootenay_nov = HALIFAX_KOOTENAY_Nov[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_stanfield_intl_a1_nov = HALIFAX_STANFIELD_INTL_A1_Nov[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_stanfield_intl_a2_nov = HALIFAX_STANFIELD_INTL_A2_Nov[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_windsor_nov = HALIFAX_WINDSOR_PARK_Nov[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]

        # dec
        halifax_dockyard_dec = HALIFAX_DOCKYARD_Dec[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_kootenay_dec = HALIFAX_KOOTENAY_Dec[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_stanfield_intl_a1_dec = HALIFAX_STANFIELD_INTL_A1_Dec[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_stanfield_intl_a2_dec = HALIFAX_STANFIELD_INTL_A2_Dec[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]
        halifax_windsor_dec = HALIFAX_WINDSOR_PARK_Dec[
            ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)',
             'Stn Press (kPa)']]

        # compute month-wise average of following columns from five different dataset sources:
        # Temp, Dew Point temp, Rel Hum , Wind Spd, Visibility, Stn Press
        avg_sep = pd.concat(
            [halifax_dockyard_sep, halifax_kootenay_sep, halifax_stanfield_intl_a1_sep, halifax_stanfield_intl_a2_sep,
             halifax_windsor_sep], ignore_index=True).groupby(['Date/Time']).mean().reset_index()

        avg_oct = pd.concat(
            [halifax_dockyard_oct, halifax_kootenay_oct, halifax_stanfield_intl_a1_oct, halifax_stanfield_intl_a2_oct,
             halifax_windsor_oct], ignore_index=True).groupby(['Date/Time']).mean().reset_index()

        avg_nov = pd.concat(
            [halifax_dockyard_nov, halifax_kootenay_nov, halifax_stanfield_intl_a1_nov, halifax_stanfield_intl_a2_nov,
             halifax_windsor_nov], ignore_index=True).groupby(['Date/Time']).mean().reset_index()

        avg_dec = pd.concat(
            [halifax_dockyard_dec, halifax_kootenay_dec, halifax_stanfield_intl_a1_dec, halifax_stanfield_intl_a2_dec,
             halifax_windsor_dec], ignore_index=True).groupby(['Date/Time']).mean().reset_index()

        # concatinate month-wise average dataset from Sep-Dec
        avg_weather_data = pd.concat([avg_sep, avg_oct, avg_nov, avg_dec])
        len(avg_weather_data)
        return avg_weather_data


    def create_work_column(self, avg_weather_data):
        # create work column (hour-wise) based on the working hours, weekends, and holidays
        work_column = pd.DataFrame(columns=['Work'])

        # Mon-fri: 9Am to 5Pm (working hours)
        # Oct 27 to Nov 3 : no work (holidays)
        start_date_of_no_work = '2019-10-27 00:00'
        end_date_of_no_work = '2019-11-03 23:00'

        for i, d in avg_weather_data['Date/Time'].iteritems():
            date_time = datetime.datetime.strptime(d, '%Y-%m-%d %H:%M')
            if date_time.weekday() in (5, 6):
                # if weekend
                work_column = work_column.append({'Work': 0}, ignore_index=True)
            elif (start_date_of_no_work <= d) and (end_date_of_no_work >= d):
                # holidays
                work_column = work_column.append({'Work': 0}, ignore_index=True)
            else:
                work_hours = d.split()[-1]
                # working hours: 9AM to 5PM
                if work_hours >= '09:00' and work_hours <= '17:00':
                    work_column = work_column.append({'Work': 1}, ignore_index=True)
                else:
                    work_column = work_column.append({'Work': 0}, ignore_index=True)

        avg_weather_data['Date/Time'].iloc[0:5], work_column[0:5]
        len(work_column)
        # add Work col to our weather dataset
        avg_weather_data['Work'] = work_column['Work']


    def create_headache_column(self, avg_weather_data):
        # store given date and time of headaches in list
        headache_dates = [['2019-09-05', '07:00', '10:00', 8], ['2019-09-15', '09:00', '15:00', 7],
                          ['2019-09-27', '15:00', '21:00', 8],
                          ['2019-10-15', '14:00', '18:00', 7], ['2019-10-16', '04:00', '18:00', 10],
                          ['2019-10-28', '15:00', '21:00', 6],
                          ['2019-11-28', '17:00', '18:00', 6], ['2019-12-10', '05:00', '10:00', 4],
                          ['2019-12-15', '02:00', '10:00', 7],
                          ['2019-12-20', '13:00', '18:00', 9]]

        # create headache column based on headache date/time
        headache_column = pd.DataFrame(columns=['Headache'])
        headache_dict = {}
        for headache_date in headache_dates:
            start_time = int(headache_date[1].split(':')[0])
            end_time = int(headache_date[2].split(':')[0])
            for hour in range(start_time, end_time + 1):
                headache_dict[f"{headache_date[0]} {hour:02}:00"] = headache_date[3]

        len(headache_dict.keys())

        for i, d in avg_weather_data['Date/Time'].iteritems():
            if headache_dict.get(d) != None:
                print(d, headache_dict.get(d))
                headache_column = headache_column.append({'Headache': 1}, ignore_index=True)
            else:
                headache_column = headache_column.append({'Headache': 0}, ignore_index=True)

        # add Headache column to our weather dataset
        avg_weather_data['Headache'] = headache_column['Headache']

    def create_workout_column(self, avg_weather_data):
        # store given workout/gym date and time in a list
        workout_dates = [['2019-09-02', '18:00', '19:00'], ['2019-09-04', '18:00', '19:00'],
                         ['2019-09-06', '18:00', '19:00'],
                         ['2019-09-27', '18:00', '19:00'],
                         ['2019-10-08', '18:00', '19:00'], ['2019-10-20', '20:00', '21:00'],
                         ['2019-10-27', '06:00', '07:00'],
                         ['2019-10-28', '18:00', '19:00'],
                         ['2019-11-02', '18:00', '19:00'], ['2019-11-07', '18:00', '19:00'],
                         ['2019-11-09', '18:00', '19:00'],
                         ['2019-11-12', '18:00', '19:00'], ['2019-11-20', '18:00', '19:00'],
                         ['2019-12-15', '18:00', '19:00'], ['2019-12-16', '18:00', '19:00']]

        # create workout/gym column (hour-wise) based on given workout/gym date/time.
        workout_column = pd.DataFrame(columns=['Workout'])
        workout_dict = {}
        for workout_date in workout_dates:
            start_time = int(workout_date[1].split(':')[0])
            end_time = int(workout_date[2].split(':')[0])
            print(start_time, end_time)
            for hour in range(start_time, end_time + 1):
                workout_dict[f"{workout_date[0]} {hour:02}:00"] = 1

        for i, d in avg_weather_data['Date/Time'].iteritems():
            if workout_dict.get(d) != None:
                print(d, workout_dict.get(d))
                workout_column = workout_column.append({'Workout': workout_dict.get(d)}, ignore_index=True)
            else:
                workout_column = workout_column.append({'Workout': 0}, ignore_index=True)

        # add Workout to our weather dataset
        avg_weather_data['Workout'] = workout_column['Workout']

    def save_dataset_to_csv(self, avg_weather_data):
        # save dataset as a csv file
        avg_weather_data.to_csv("full_dataset.csv", index=False)


if __name__ == "__main__":
    dataset_prep = DatasetPrep()
    avg_weather_data = dataset_prep.read_and_prepare_weather_dataset()
    dataset_prep.create_work_column(avg_weather_data)
    dataset_prep.create_workout_column(avg_weather_data)
    dataset_prep.create_headache_column(avg_weather_data)
    dataset_prep.save_dataset_to_csv(avg_weather_data)
    print("ok")
