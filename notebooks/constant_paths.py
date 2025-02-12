prefix =  "/home/geoffroy/Documents/BrisbaneDataset/"

nmea_paths = {
    'sunset1': prefix + "sunset1/20200421_170039-sunset1_concat.nmea",
    'sunset2': prefix + "sunset2/20200422_172431-sunset2_concat.nmea",
    'daytime': prefix + "daytime/20200424_151015-daytime_concat.nmea",
    'morning': prefix + "morning/20200428_091154-morning_concat.nmea",
    'sunrise': prefix + "sunrise/20200429_061912-sunrise_concat.nmea",
    'night': prefix + "night/20200427_181204-night_concat.nmea"
}

video_paths = {
    'sunset1': prefix + "sunset1/20200421_170039-sunset1_concat.mp4",
    'sunset2': prefix + "sunset2/20200422_172431-sunset2_concat.mp4",
    'daytime': prefix + "daytime/20200424_151015-daytime_concat.mp4",
    'morning': prefix + "morning/20200428_091154-morning_concat.mp4",
    'sunrise': prefix + "sunrise/20200429_061912-sunrise_concat.mp4",
    'night': prefix + "night/20200427_181204-night_concat.mp4"
}

event_paths = {
   'sunset1': "../data/dvs_vpr_2020-04-21-17-03-03.zip",
    'sunset2': "../data/dvs_vpr_2020-04-22-17-24-21.zip",
    'daytime': "../data/dvs_vpr_2020-04-24-15-12-03.zip",
    'morning': "../data/dvs_vpr_2020-04-28-09-14-11.zip",
    'sunrise': "../data/dvs_vpr_2020-04-29-06-20-23.zip",
    'night': "../data/dvs_vpr_2020-04-27-18-13-29.zip"
}

frame_paths = {
    'sunset1': prefix + "sunset1/frames/",
    'sunset2': prefix + "sunset2/frames/",
    'daytime': prefix + "daytime/frames/",
    'morning': prefix + "morning/frames/",
    'sunrise': prefix + "sunrise/frames/",
    'night': prefix + "night/frames/"
}



video_beginning = {
    'sunset1': 1587452582.35,
    'sunset2': 1587540271.65,
    'daytime': 1587705130.80,
    'morning': 1588029265.73,
    'sunrise': 1588105232.91,
    'night': 1587975221.10
}