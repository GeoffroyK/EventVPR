import math
import pynmea2
import numpy as np

# Taken from https://github.com/Tobias-Fischer/ensemble-event-vpr/blob/master/read_gps.py
def get_gps(nmea_file_path):
    nmea_file = open(nmea_file_path, encoding='utf-8')

    latitudes, longitudes, timestamps = [], [], []

    first_timestamp = None
    previous_lat, previous_lon = 0, 0

    for line in nmea_file.readlines():
        try:
            msg = pynmea2.parse(line)
            if first_timestamp is None:
                first_timestamp = msg.timestamp
            if msg.sentence_type not in ['GSV', 'VTG', 'GSA']:
                # print(msg.timestamp, msg.latitude, msg.longitude)
                # print(repr(msg.latitude))
                dist_to_prev = np.linalg.norm(np.array([msg.latitude, msg.longitude]) - np.array([previous_lat, previous_lon]))
                if msg.latitude != 0 and msg.longitude != 0 and msg.latitude != previous_lat and msg.longitude != previous_lon and dist_to_prev > 0.0001:
                    timestamp_diff = (msg.timestamp.hour - first_timestamp.hour) * 3600 + (msg.timestamp.minute - first_timestamp.minute) * 60 + (msg.timestamp.second - first_timestamp.second)
                    latitudes.append(msg.latitude); longitudes.append(msg.longitude); timestamps.append(timestamp_diff)
                    previous_lat, previous_lon = msg.latitude, msg.longitude

        except pynmea2.ParseError as e:
            # print('Parse error: {} {}'.format(msg.sentence_type, e))
            continue

    return np.array(np.vstack((latitudes, longitudes, timestamps))).T

def getDistanceFromLatLongKm(lat1, lon1, lat2, lon2):
    '''
    Distance calculation between lat/long coordinates
    Haversine Formula
    with earth_radius = 6371km
    '''
    EARTH_RADIUS = 6371.

    #Convert lat long from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = EARTH_RADIUS * c
    return distance

def sampleItineraryFromDistance(latitudes: np.array, longitudes: np.array, samplingDistance:float) -> list:
    '''
    Sample itinerary points at a fixed sampling distance calculated through Haversine
    Returns list of sampled points
    '''
    index = 0
    distance = 0
    sampledPoints = []

    a = np.array([latitudes[0], longitudes[0]]) # Reference point
    b = np.array([latitudes[index], longitudes[index]])

    while index < len(latitudes) - 1: # While all the coordinates are not computed 
        while distance < samplingDistance: # Calculate the point b index where distance(a,b)=samplingDistance
            b = np.array([latitudes[index], longitudes[index]])
            distance = getDistanceFromLatLongKm(a[0], a[1], b[0], b[1])
                    
            if index < len(latitudes) - 1:
                index += 1
            else:
                print(f"Breaking at {index}")
                break
        a = b # Compute for the next point
        distance = 0
        sampledPoints.append(index)
    return sampledPoints