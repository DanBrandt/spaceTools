# This module contains various codes for downloading space weather data from open-source repositories.
# ----------------------------------------------------------------------------------------------------------------------
# Top-level Imports:
import os
import csv
import pathlib
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pooch
from netCDF4 import Dataset
from tqdm import tqdm
# ----------------------------------------------------------------------------------------------------------------------
# Directory Management:
here = pathlib.Path(__file__).parent.resolve()
data_directory = pathlib.Path(__file__).parent.parent.parent.resolve() / 'data'
# ----------------------------------------------------------------------------------------------------------------------
# Constants
h = 6.62607015e-34 # Planck's constant in SI units of J s
c = 299792458 # Speed of light in m s^-1
# ----------------------------------------------------------------------------------------------------------------------
# Helper Functions
def urlObtain(URL, loc=None, fname=None, hash=None):
    """
    Helper function that uses Pooch to download files to a location specified by the user.
    :param URL: str
        The location of a file to be downloaded.
    :param loc: str
        The place the file will be downloaded.
    :param fname: str
        The name the file will have once it is downloaded.
    :param hash: str
        A known hash (checksum) of the file. Will be used to verify the download or check if an existing file needs to
        be updated.
    :return:
    """
    if loc is None:
        loc = os.getcwd()
    if os.path.isfile(str(loc)+'/'+fname) is False:
        fname_loc = pooch.retrieve(url=URL, known_hash=hash, fname=fname, path=loc)
    else:
        fname_loc = str(loc)+'/'+fname

    return fname_loc

def numberStr(num, zeroes=1):
    """
    Takes an integer and converts it to a string with the number of zeros in front of it determined by the user.
    :param num: int
        A given integer.
    :param zeroes: int
        The number of zeroes to put in front of the number.
    :return numStr: str
        The string of the number with the desired number of zeroes in front of it.
    """
    myStr = ''
    for i in range(zeroes):
        myStr += '0'
    numStr = myStr+str(num)

    return numStr

def combine_time_series(time_series_1, time_series_2, priority=1):
    """
    Take two time series and combine them into a single time series. If there are overlapping regions in the time
    series, selectively use the values in one of the time series only, according to what the user states.
    :param time_series_1: list
        List contains two elements. The first contains a numpy array of datetimes and the second contains a numpy array
        of values of the dependent variables.
    :param time_series_2: list
        List with the same format as time_series_1, but for a second time series.
    :param priority: int
        Signals which time series should have priority when there are overlapping regions.
    :return times: numpy.ndarray
        An array of datetimes for the time series data.
    :return values: numpy.ndarray
        An array of values for the time series data.
    """
    # Unpack the inputs:
    first_times, first_values = time_series_1
    second_times, second_values = time_series_2

    # Find regions of overlap:
    out, comm1, comm2 = np.intersect1d(first_times, second_times, return_indices=True)

    # Isolate the overlapping values:
    first_overlap = first_values[comm1]
    second_overlap = second_values[comm2]
    overlap_values = [first_overlap, second_overlap]

    # Choose the values from the time series the user wants:
    chosen_values = overlap_values[priority]
    first_values[comm1] = chosen_values
    second_values[comm2] = chosen_values

    # Reconstruct the combined time series (with the desired values in the overlapping region):
    limit = np.where(first_times == first_times[comm1][0])[0][0]
    first_times_clipped = first_times[:limit]
    first_values_clipped = first_values[:limit]
    times = np.concatenate((first_times_clipped, second_times))
    values = np.concatenate((first_values_clipped, second_values))

    return times, values

def read_dcx_file(fname):
    """
    Open and read a .txt file containing DCX data downloaded from the University of Oulu website.
    :param fname: str
        The filename to load in.
    :return times: numpy.ndarray
        An array of datetimes for each DCX observation.
    :return dcx: numpy.ndarray
        A 1D-array of DCX values.
    """
    with open(fname, 'r') as open_file:
        fileInfo = open_file.readlines()
        times = np.zeros(len(fileInfo), dtype=object)
        dcx = np.zeros(len(fileInfo), dtype=float)
        for i in range(len(fileInfo)):
            parsed_line = fileInfo[i].split()
            current_time = datetime(int(parsed_line[1]), int(parsed_line[2]), int(parsed_line[3]), int(parsed_line[4]))
            current_dcx = float(parsed_line[-1])
            times[i] = current_time
            dcx[i] = current_dcx
    return times, dcx

def getIrr(dateStart: str, dateEnd: str, source: str, downloadDir=None):
    """
    Given start date and end date, automatically download irradiance data
    Data from LISIRD, source specified by user
    Includes FISM2 (daily or stan bands) or SEE (Level 3 daily).
    :param dateStart: str
        The starting date for the data in YYYY-MM-DD format.
    :param dateEnd: str
        The ending date for the data in YYYY-MM-DD format.
    :param source: str
        The type of data to be obtained. Valid inputs are:
        - FISM2 (for daily averages of FISM2 data)
        - FISM2S (for daily averages of FISM2 standard bands)
            According to Solomon and Qian 2005
        - SEE (for Level 3 daily averages of TIMED/SEE data)
    :param downloadDir: str, optional
        Location FISM2 data will be saved.
        Default=none, saved to top directory package is in
    :return times: ndarray
        Datetime values for each spectrum.
    :return wavelengths: ndarray
        Wavelength bins (bin boundaries) for the spectral data.
    :return irradiance: ndarray
        2D array: each row is a spectrum at a particular time
            columns are wavelength bands
    :return uncertainties: ndarray
        2D array: each row is a different time
            columns are uncertainties for each wavelength band
    """
    # Converting the input time strings to datetimes:
    dateStartDatetime = datetime.strptime(dateStart, "%Y-%m-%d")
    dateEndDatetime = datetime.strptime(dateEnd, "%Y-%m-%d")

    # Check if the user has asked for a source that can be obtained:
    validSources = ['FISM2', 'FISM2S', 'SEE']
    if source not in validSources:
        raise ValueError("Variable 'source' must be either"
                         + " 'FISM2', 'FISM2S', or 'SEE'.")

    # If download directory not specified, set to top directory package is in:
    if downloadDir is None:
        downloadDir = here

    # Download most recent file for the corresponding source and read it in:
    if source == 'FISM2':
        url = 'https://lasp.colorado.edu/eve/data_access/eve_data/fism/daily_hr_data/daily_data.nc'
        fname = 'FISM2_daily_data.nc'
        if os.path.isfile(pathlib.Path(downloadDir).joinpath(fname)) == False:
            urlObtain(url, loc=downloadDir[:-1], fname=fname)
        datetimes, wavelengths, irradiance, uncertainties =\
            obtainFism2(pathlib.Path(downloadDir).joinpath(fname))
    elif source == 'FISM2S':
        url = 'https://lasp.colorado.edu/eve/data_access/eve_data/fism/daily_bands/daily_bands.nc'
        fname = 'FISM2_daily_bands.nc'
        if os.path.isfile(pathlib.Path(downloadDir).joinpath(fname)) == False:
            urlObtain(url, loc=downloadDir[:-1], fname=fname)
        datetimes, wavelengths, irradiance, uncertainties =\
            obtainFism2(pathlib.Path(downloadDir).joinpath(fname), bands=True)
    else:
        url = 'https://lasp.colorado.edu/data/timed_see/level3/latest_see_L3_merged.ncdf'
        fname = 'TIMED_SEE_Level_3.nc'
        if os.path.isfile(pathlib.Path(downloadDir).joinpath(fname)) == False:
            urlObtain(url, loc=downloadDir[:-1], fname=fname)
        datetimes, wavelengths, irradiance, uncertainties =\
            obtainSEE(pathlib.Path(downloadDir).joinpath(fname) + fname)

    # Subset the data according to user demands:
    validInds = np.where((datetimes >= dateStartDatetime)
                         & (datetimes <= dateEndDatetime))[0]
    times = datetimes[validInds]
    if source == 'FISM2S':
        irradiance = irradiance[-1,validInds,:]
        uncertainties = uncertainties[validInds,:]
    else:
        irradiance = irradiance[validInds,:]
        uncertainties = uncertainties[validInds,:]

    # Return the resulting data:
    return times, wavelengths, irradiance, uncertainties

def obtainFism2(myFism2File: str, bands=False):
    """
    Load in spectrum data from a FISM2 file.
    :param myFism2File: str
        The location of the NETCDF4 file.
    :param bands: bool, optional
        If True, loads in standard segmented data
            data segmented into Solomon and Qian 2005 standard bands.
        Default: False
    :return datetimes: ndarray
        An array of datetimes for each TIMED/SEE spectra.
    :return wavelengths: ndarray
        1d array of wavelengths at which there are irradiance values.
    :return irradiances: ndarray
        2d array of irradiance values at each time.
    :return uncertainties: ndarray
        2d array of irradiance uncertainty values at each time.
    """
    fism2Data = Dataset(myFism2File)
    wavelengths = np.asarray(fism2Data.variables["wavelength"])
    if bands == True:  # STANDARD BANDS
        flux = np.asarray(fism2Data.variables["ssi"])  # photons/cm2/second
        # bandwidths = np.asarray(fism2Data.variables['band_width'])
        pFlux = flux * 1.0e4  # photons/m2/second
        # Convert fluxes to irradiances:
        irr = np.zeros_like(flux)
        for i in tqdm(range(flux.shape[1])):
            irr[:,i] = spectralIrradiance(pFlux[:,i], wavelengths[i] * 10.0)  # W/m^2
        irradiance = np.array([flux, irr])
        uncertainties = np.full_like(irradiance, fill_value=np.nan) # TODO: Replace with an estimation of uncertainty
    else:  # NATIVE DATA
        irradiance = np.asarray(fism2Data.variables["irradiance"])  # W/m^2/nm
        uncertainties = np.asarray(fism2Data.variables["uncertainty"])
    dates = fism2Data.variables["date"]
    datetimes = []
    for i in tqdm(range(len(dates))):
        year = dates[i][:4]
        day = dates[i][4:]
        currentDatetime = (
            datetime(int(year), 1, 1)
            + timedelta(int(day)-1)
            + timedelta(hours=12)
        )
        datetimes.append(currentDatetime)
    datetimes = np.asarray(datetimes)
    return datetimes, wavelengths, irradiance, uncertainties

def obtainSEE(seeFile: str):
    """
    Given a TIMED/SEE NETCDF4 file, load in and return the timestamps, wavelengths, irradiances, and uncertainties.
    :param seeFile: str
        The NETCDF4 file containing TIMED/SEE data.
    :return datetimes: ndarray
        An array of datetimes for each TIMED/SEE spectra.
    :return wavelengths: ndarray
        A one-dimensional array of wavelengths at which there are irradiance values.
    :return irradiances: ndarray
        A two-dimensional array of irradiance values at each time.
    :return uncertainties: ndarray
        A two-dimensional array of irradiance uncertainty values at each time.
    """
    seeData = Dataset(seeFile)
    dates = np.squeeze(seeData.variables['DATE'])
    wavelengths = np.squeeze(seeData.variables['SP_WAVE'])
    irradiances = np.squeeze(seeData.variables['SP_FLUX'])
    uncertainties = np.squeeze(seeData.variables['SP_ERR_TOT'])
    precision = np.squeeze(seeData.variables['SP_ERR_MEAS'])
    datetimes = []
    for i in range(len(dates)):
        year = str(dates[i])[:4]
        day = str(dates[i])[4:]
        currentDatetime = (datetime(int(year), 1, 1)
                           + timedelta(int(day)-1) + timedelta(hours=12))
        datetimes.append(currentDatetime)
    datetimes = np.asarray(datetimes)
    return datetimes, wavelengths, irradiances, uncertainties

def spectralIrradiance(photonFlux, wavelength):
    """
    Convert the photon flux to the corresponding spectral irradiance,
    given a specific wavelength.
    :param: photonFlux: ndarray, float, or int
        Photon flux in units of photons s^-1 m^-2.
        For a singular wavelength, units are photons m^-2
    :param: wavelength: float
        A specific wavelength in Angstroms.
    :return: irradiance: ndarray or float
        The corresponding spectral irradiance in units of W/m^2/nm.
    """
    # Convert the wavelength in the denominator to meters.
    photonEnergy = (h*c) / (wavelength*1e-10)
    irradiance = photonFlux * photonEnergy
    return irradiance
# ----------------------------------------------------------------------------------------------------------------------
# Main Functions
def get_dcx(cutoff_date=None):
    """
    Downloads the most recent DCX index data from the University of Oulu's online repository. Note that this function
    downloads 4-station Dcx data extending from August 3, 1932 to December 31, 2016, and combines it with 14-station
    Dcx data extending from January 1, 2015 to the most recent date (typically 3 months before the present date). In the
    period overlapping coverage for both quantities, the 14-station data is used to overwrite the 4-station data. The
    combined data is saved to a .csv file, with the first column being time stamps and the second column being Dcx
    values in nT.

    Note that 4-station data is obtained from: http://dcx.oulu.fi/?link=queryDefinite&a=plotti&yI=1932&mI=08&dI=03&hI=00&yF=2016&mF=12&dF=31&hF=23&res=h&index=Dcx&type=
    Note that 14-station data is obtained from: http://dcx.oulu.fi/?link=queryProvisional&a=plotti&yI=2015&mI=01&dI=01&hI=00&yF=2025&mF=03&dF=07&hF=12&res=h&index=Dcx&type=14

    :param cutoff_date: str
        An optional date after which to not include any data. Must be in YYYY-MM-DD.
    :return dcx_filename: str
        The name (location) of the file that was created from the downloaded data.
    """
    # If the data already has been downloaded, just load it in:
    dcx_filename = str(data_directory) + '/dcx_data_1932-08-03_'+cutoff_date+'.csv' # + str(dcx_times[0])[:10] + '_' + str(dcx_times[-1])[:10] + '.csv'
    if os.path.isfile(dcx_filename) == True:
        with open(dcx_filename, newline='') as csvfile:
            data = list(csv.reader(csvfile))
        day_strings = [element[0].split()[0] for element in data[1:]]
        hour_strings = [element[0].split()[1] for element in data[1:]]
        dcx_values = np.array([float(element[0].split()[2]) for element in data[1:]])
        dcx_times = np.array([datetime.strptime(x+'T'+y, "%Y-%m-%dT%H:%M:%S") for x,y in zip(day_strings, hour_strings)])
    else:
        # Query the University of Oulu website to obtain the raw data:
        four_station_filename = urlObtain('http://dcx.oulu.fi/tmp/Dcxh3208030016123123.txt', loc=data_directory, fname='Dcxh3208030016123123.txt', hash='f6487bb1da1862b56257da76652410f1320cb9faec20c8382b39023b536418d2')
        today = datetime.strptime(datetime.today().strftime('%Y-%m-%d'), '%Y-%m-%d') + timedelta(hours=12)
        most_recent_date_for_valid_data = today - timedelta(days=int(3*30))
        if cutoff_date is None:
            yearStr = str(most_recent_date_for_valid_data.year)[-2:]
            monthStr = numberStr(most_recent_date_for_valid_data.month)
            dayStr = numberStr(most_recent_date_for_valid_data.day)
        else:
            yearStr = cutoff_date[2:4]
            monthStr = cutoff_date[5:7]
            dayStr = cutoff_date[8:10]
        dcx_14_station_file_str = 'Prov.Dcx14h15010100'+yearStr+monthStr+dayStr+'12.txt'
        fourteen_station_filename = urlObtain('http://dcx.oulu.fi/tmp/'+dcx_14_station_file_str, loc=data_directory, fname=dcx_14_station_file_str)

        # Read the data from the downloaded files:
        four_station_times, four_station_dcx = read_dcx_file(four_station_filename)
        fourteen_station_times, fourteen_station_dcx = read_dcx_file(fourteen_station_filename)

        # Combine the data together into a single time series:
        time_series_1 = [four_station_times, four_station_dcx]
        time_series_2 = [fourteen_station_times, fourteen_station_dcx]
        dcx_times, dcx_values = combine_time_series(time_series_1, time_series_2)

        # Make a plot for a sanity check:
        plt.figure(figsize=(15, 8))
        plt.plot(dcx_times, dcx_values)
        plt.xlabel('Time')
        plt.ylabel('Dcx (nT)')

        # Put the raw data into a .csv file and save it out:
        dcx_filename = str(data_directory) + '/dcx_data_'+str(dcx_times[0])[:10]+'_'+str(dcx_times[-1])[:10]+'.csv'
        with open(dcx_filename, 'w') as file:
            file.write("Date Time DCX\n")
            for i in range(len(dcx_times)):
                line = str( dcx_times[i] )+" "+str(dcx_values[i])+"\n"
                file.write(line)
        print('DCX data saved to '+dcx_filename)

    return dcx_times, dcx_values