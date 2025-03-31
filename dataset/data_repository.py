
DATABASE_LOCATION = "Mekong_Data"
ORIGINAL_DATA = "Original_Data"
FILLED_GAPS_DATA = "Filled_Gaps_Data"
SMOOTHED_FILLED_GAPS_DATA = "Smoothed_Filled_Gaps_Data"
WATER_LEVEL_CATEGORY = 'Water.Level'
RAINFALL_CATEGORY = 'Rainfall.Manual'
WATER_DISCHARGE_CATEGORY = 'Discharge.Daily'
MAIN_FOCUS_CATEGORY = 'Discharge.Daily'
STORAGE_LOCATION = "Storage"
PEAK_STATIONS_JSON = "peaks_stations.json"
ROOT_STATION_IN_MAIN_WATER_FLOW_LIST = "Chiang Saen"
HIERARCHY_STATIONS_FLOW_TREE = "hierarchy_stations_flow_tree"
 
CATEGORY_PATHS = {
    'Water Level (Original)': '{DATABASE_LOCATION}/{ORIGINAL_DATA}/Water.Level/',
    'Water Level (Filled_Gaps)': '{DATABASE_LOCATION}/{FILLED_GAPS_DATA}/Water.Level/',
    'Water Level (Smoothed_Filled_Gaps)': '{DATABASE_LOCATION}/{SMOOTHED_FILLED_GAPS_DATA}/Water.Level/'
}
 
COMMON_DURATION_LIST = {'Water.Level': {'start_duration': '1989-01-01', 'end_duration': '2002-10-31'}, 
                        'Rainfall.Manual': {'start_duration': '1989-01-01', 'end_duration': '2002-10-31'},
                        'Discharge.Daily': {'start_duration': '1989-01-01', 'end_duration': '2002-10-31'}}

TRAINING_RANGE_LIST = { # how many years for training
  'Discharge.Daily': 9,
  'Rainfall.Manual': 9, 
  'Water.Level': 9
}

VALIDATION_RANGE_LIST = { # how many years for validation
  'Discharge.Daily': 2,
  'Rainfall.Manual': 2,
  'Water.Level': 2
}

# Parameters for peak detection
PROMINENCE = 1.0  # Minimum prominence for peaks
DISTANCE = 10      # Minimum distance between peaks (in days)
WIDTH = 7          # Minimum width of peaks (smooth minor fluctuations)
MAX_TIME_LAG_THRESHOLD = 30  # Max difference (in days) to match peaks between stations

MAIN_STATIONS_LIST = ['Kompong Cham', 'Chaktomuk', 'Vientiane KM4', 'Ban Pak Kanhoung', 'Ban Na Luang', 'Chiang Saen', 'Ban Huai Yano Mai', 'Ban Tha Ton', 'Ban Tha Mai Liam', 'Ban Pak Huai', 'Yasothon', 'Ban Chot', 'Ban Nong Kiang', 'Ban Tad Ton', 'Ban Huai Khayuong', 'Cau 14 (Buon Bur)', 'Stung Treng', 'Pakse', 'Ban Kengdone', 'Chiang Khan', 'Nong Khai', 'Nakhon Phanom', 'Mukdahan', 'Khong Chiam', 'Kontum', 'Duc Xuyen']

TOTAL_WATER_FLOW_LIST = {
  "Jinghong": ["Chiang Saen"],
  "Chiang Saen": ["Ban Huai Yano Mai", "Ban Tha Ton", "Ban Tha Mai Liam", "Luang Prabang"],
  "Ban Huai Yano Mai": [],
  "Ban Tha Ton": [],
  "Ban Tha Mai Liam": [],
  "Luang Prabang": ["Ban Pak Huai", "Chiang Khan"],
  "Ban Pak Huai": [],
  "Chiang Khan": ["Vientiane KM4"],
  "Vientiane KM4": ["Nong Khai"],
  "Nong Khai": ["Ban Pak Kanhoung", "Nakhon Phanom"],
  "Ban Pak Kanhoung": ["Ban Na Luang"],
  "Ban Na Luang": [],
  "Nakhon Phanom": ["Mukdahan"],
  "Mukdahan": ["Ban Kengdone", "Khong Chiam"],
  "Ban Kengdone": [],
  "Khong Chiam": ["Yasothon", "Pakse"],
  "Yasothon": ["Ban Chot"],
  "Ban Chot": ["Ban Tad Ton", "Ban Nong Kiang"],
  "Ban Tad Ton": [],
  "Ban Nong Kiang": [],
  "Pakse": ["Ban Huai Khayuong", "Stung Treng"],
  "Ban Huai Khayuong": [],
  "Stung Treng": ["Chantangoy", "Ban Kamphun", "Kratie"],
  "Chantangoy": [],
  "Ban Kamphun": ["Kontum", "Dak To", "Pleiku", "Lumphat"],
  "Kontum": [],
  "Dak To": [],
  "Pleiku": [],
  "Lumphat": ["Ban Don"],
  "Ban Don": ["Cau 14 (Buon Bur)", "Buon Me Thuoc"],
  "Cau 14 (Buon Bur)": ["Duc Xuyen", "Giang Son"],
  "Duc Xuyen": ["Dak Nong"],
  "Dak Nong": [],
  "Giang Son": ["Buon Ho"],
  "Buon Ho": [],
  "Buon Me Thuoc": [],
  "Kratie": ["Kompong Cham"],
  "Kompong Cham": ["Chroy Chang Var", "Ben Luc"],
  "Chroy Chang Var": ["Chaktomuk", "Tan Chau", "Chau Doc"],
  "Chaktomuk": ["Phnom Penh Port", "Battambang"],
  "Phnom Penh Port": [],
  "Battambang": [],
  "Tan Chau": ["Chau Doc", "Vam Nao", "Cho Moi", "My Thuan"],
  "Vam Nao": ["Cho Moi", "Can Tho", "Tan Hiep"],
  "Chau Doc": ["Can Tho", "Tan Hiep"],
  "My Thuan": ["Cai Be", "Cho Lach"],
  "Cai Be": ["Cai Lay"],
  "Cai Lay": [],
  "Cho Lach": ["My Tho", "Batri"],
  "My Tho": ["Vam Kenh"],
  "Vam Kenh": [],
  "Batri": [],
  "Can Tho": ["Vi Thanh", "Ca Mau", "Dai Ngai"],
  "Vi Thanh": ["Phung Hiep"],
  "Phung Hiep": [],
  "Ca Mau": [],
  "Dai Ngai": ["Dinh An"],
  "Dinh An": [],
  "Ben Luc": []
}

MAIN_WATER_FLOW_LIST = {
    "Ban Don": [
        "Duc Xuyen"
    ],
    "Ban Kengdone": [],
    "Ban Pak Kanhoung": [],
    "Can Tho": [
        "Phung Hiep"
    ],
    "Chaktomuk": [
        "Phnom Penh Port"
    ],
    "Chau Doc": [
        "Can Tho"
    ],
    "Chiang Khan": [
        "Vientiane KM4"
    ],
    "Chiang Saen": [
        "Luang Prabang"
    ],
    "Duc Xuyen": [],
    "Khong Chiam": [
        "Pakse"
    ],
    "Kompong Cham": [
        "Chaktomuk",
        "Tan Chau",
        "Chau Doc"
    ],
    "Kontum": [],
    "Kratie": [
        "Kompong Cham"
    ],
    "Luang Prabang": [
        "Chiang Khan"
    ],
    "Lumphat": [
        "Ban Don"
    ],
    "Mukdahan": [
        "Ban Kengdone",
        "Khong Chiam"
    ],
    "My Thuan": [
        "Vam Kenh"
    ],
    "Nakhon Phanom": [
        "Mukdahan"
    ],
    "Nong Khai": [
        "Ban Pak Kanhoung",
        "Nakhon Phanom"
    ],
    "Pakse": [
        "Stung Treng"
    ],
    "Phnom Penh Port": [],
    "Phung Hiep": [],
    "Stung Treng": [
        "Kontum",
        "Lumphat",
        "Kratie"
    ],
    "Tan Chau": [
        "Chau Doc",
        "Vam Nao",
        "My Thuan"
    ],
    "Vam Kenh": [],
    "Vam Nao": [
        "Can Tho"
    ],
    "Vientiane KM4": [
        "Nong Khai"
    ]
}

LAGS_AND_DISTANCE_LIST = [
    {
        "station_1": "Chiang Saen",
        "station_2": "Luang Prabang",
        "distance_km": 218.07920604294202,
        "lags_days": 1.1296296296296295
    },
    {
        "station_1": "Luang Prabang",
        "station_2": "Chiang Khan",
        "distance_km": 225.91447029260135,
        "lags_days": 3.6451612903225805
    },
    {
        "station_1": "Chiang Khan",
        "station_2": "Vientiane KM4",
        "distance_km": 100.2563652951395,
        "lags_days": 1.5833333333333333
    },
    {
        "station_1": "Vientiane KM4",
        "station_2": "Nong Khai",
        "distance_km": 13.520816175484997,
        "lags_days": 0.543859649122807
    },
    {
        "station_1": "Nong Khai",
        "station_2": "Ban Pak Kanhoung",
        "distance_km": 62.70356855985651,
        "lags_days": 0.5789473684210527
    },
    {
        "station_1": "Nong Khai",
        "station_2": "Nakhon Phanom",
        "distance_km": 222.44885912205206,
        "lags_days": 1.0909090909090908
    },
    {
        "station_1": "Nakhon Phanom",
        "station_2": "Mukdahan",
        "distance_km": 93.3475111045376,
        "lags_days": -0.42
    },
    {
        "station_1": "Mukdahan",
        "station_2": "Ban Kengdone",
        "distance_km": 75.83349337093162,
        "lags_days": 0.2647058823529412
    },
    {
        "station_1": "Mukdahan",
        "station_2": "Khong Chiam",
        "distance_km": 161.51675672971123,
        "lags_days": 1.6862745098039216
    },
    {
        "station_1": "Khong Chiam",
        "station_2": "Pakse",
        "distance_km": 42.251608388622834,
        "lags_days": 0.36666666666666664
    },
    {
        "station_1": "Pakse",
        "station_2": "Stung Treng",
        "distance_km": 174.0338858452312,
        "lags_days": 1.56
    },
    {
        "station_1": "Stung Treng",
        "station_2": "Kontum",
        "distance_km": 242.56725828581193,
        "lags_days": 0.5714285714285714
    },
    {
        "station_1": "Stung Treng",
        "station_2": "Lumphat",
        "distance_km": 110.5808621565578,
        "lags_days": -1.0
    },
    {
        "station_1": "Stung Treng",
        "station_2": "Kratie",
        "distance_km": 116.5125493159572,
        "lags_days": 1.490566037735849
    },
    {
        "station_1": "Lumphat",
        "station_2": "Ban Don",
        "distance_km": 110.43846646093206,
        "lags_days": 8.285714285714286
    },
    {
        "station_1": "Ban Don",
        "station_2": "Duc Xuyen",
        "distance_km": 69.72301489209265,
        "lags_days": -2.3529411764705883
    },
    {
        "station_1": "Kratie",
        "station_2": "Kompong Cham",
        "distance_km": 93.4599126756562,
        "lags_days": 1.9795918367346939
    },
    {
        "station_1": "Kompong Cham",
        "station_2": "Chaktomuk",
        "distance_km": 62.25301641911656,
        "lags_days": 8.5
    },
    {
        "station_1": "Kompong Cham",
        "station_2": "Tan Chau",
        "distance_km": 123.72071084104947,
        "lags_days": 13.181818181818182
    },
    {
        "station_1": "Kompong Cham",
        "station_2": "Chau Doc",
        "distance_km": 136.14849950516464,
        "lags_days": 14.952380952380953
    },
    {
        "station_1": "Chaktomuk",
        "station_2": "Phnom Penh Port",
        "distance_km": 1.7664294574218191,
        "lags_days": 0.10714285714285714
    },
    {
        "station_1": "Tan Chau",
        "station_2": "Chau Doc",
        "distance_km": 16.373276618064097,
        "lags_days": 2.576923076923077
    },
    {
        "station_1": "Tan Chau",
        "station_2": "Vam Nao",
        "distance_km": 27.606079128835457,
        "lags_days": 4.115384615384615
    },
    {
        "station_1": "Tan Chau",
        "station_2": "My Thuan",
        "distance_km": 94.27699754948245,
        "lags_days": 14.727272727272727
    },
    {
        "station_1": "Chau Doc",
        "station_2": "Can Tho",
        "distance_km": 101.640353434691,
        "lags_days": 14.434782608695652
    },
    {
        "station_1": "Can Tho",
        "station_2": "Phung Hiep",
        "distance_km": 27.13548369077115,
        "lags_days": 0.75
    },
    {
        "station_1": "Vam Nao",
        "station_2": "Can Tho",
        "distance_km": 74.40669314682741,
        "lags_days": 12.0
    },
    {
        "station_1": "My Thuan",
        "station_2": "Vam Kenh",
        "distance_km": 88.82145852280567,
        "lags_days": -5.764705882352941
    }
]

STATION_LOCATION_LIST = {
    "Stung Treng": {
        "latitude": 13.5325002670288,
        "longitude": 105.950187683105
    },
    "Kratie": {
        "latitude": 12.4814100265503,
        "longitude": 106.017616271973
    },
    "Chroy Chang Var": {
        "latitude": 11.5874,
        "longitude": 104.93842
    },
    "Kompong Cham": {
        "latitude": 11.91098617,
        "longitude": 105.3841017
    },
    "Phnom Penh Port": {
        "latitude": 11.57641,
        "longitude": 104.92651
    },
    "Chaktomuk": {
        "latitude": 11.5629901885986,
        "longitude": 104.935287475586
    },
    "Chantangoy": {
        "latitude": 13.56586178,
        "longitude": 106.0529906
    },
    "Ban Kamphun": {
        "latitude": 13.53586422,
        "longitude": 106.0519912
    },
    "Lumphat": {
        "latitude": 13.500880241394,
        "longitude": 106.971153259277
    },
    "Battambang": {
        "latitude": 13.0920000076294,
        "longitude": 103.20027923584
    },
    "Luang Prabang": {
        "latitude": 19.8927993774414,
        "longitude": 102.134178161621
    },
    "Vientiane KM4": {
        "latitude": 17.930980682373,
        "longitude": 102.615562438965
    },
    "Pakse": {
        "latitude": 15.099760055542,
        "longitude": 105.813186645508
    },
    "Ban Pak Kanhoung": {
        "latitude": 18.41937801,
        "longitude": 102.5463588
    },
    "Ban Na Luang": {
        "latitude": 18.91433689,
        "longitude": 102.7743174
    },
    "Ban Kengdone": {
        "latitude": 16.1872692108154,
        "longitude": 105.312866210938
    },
    "Chiang Saen": {
        "latitude": 20.2741203308105,
        "longitude": 100.08854675293
    },
    "Chiang Khan": {
        "latitude": 17.900260925293,
        "longitude": 101.669891357422
    },
    "Nong Khai": {
        "latitude": 17.8814392089844,
        "longitude": 102.732200622559
    },
    "Nakhon Phanom": {
        "latitude": 17.4253692626953,
        "longitude": 104.773933410645
    },
    "Mukdahan": {
        "latitude": 16.582799911499,
        "longitude": 104.733177185059
    },
    "Khong Chiam": {
        "latitude": 15.3220901489258,
        "longitude": 105.493476867676
    },
    "Ban Huai Yano Mai": {
        "latitude": 20.11316588,
        "longitude": 99.78170233
    },
    "Ban Tha Ton": {
        "latitude": 20.06116311,
        "longitude": 99.35976227
    },
    "Ban Tha Mai Liam": {
        "latitude": 20.02116689,
        "longitude": 99.35476379
    },
    "Ban Pak Huai": {
        "latitude": 17.70442284,
        "longitude": 101.4115267
    },
    "Yasothon": {
        "latitude": 15.7836604,
        "longitude": 105.1380685
    },
    "Ban Chot": {
        "latitude": 16.10158785,
        "longitude": 102.5734004
    },
    "Ban Nong Kiang": {
        "latitude": 16.13456982,
        "longitude": 101.6635215
    },
    "Ban Tad Ton": {
        "latitude": 15.94359283,
        "longitude": 102.0264762
    },
    "Ban Huai Khayuong": {
        "latitude": 15.00673518,
        "longitude": 105.6340189
    },
    "Tan Chau": {
        "latitude": 10.8006200790405,
        "longitude": 105.248016357422
    },
    "My Thuan": {
        "latitude": 10.2753200531006,
        "longitude": 105.926322937012
    },
    "My Tho": {
        "latitude": 10.35912163,
        "longitude": 106.3529997
    },
    "Cho Moi": {
        "latitude": 10.54709584,
        "longitude": 105.4561108
    },
    "Chau Doc": {
        "latitude": 10.7052803039551,
        "longitude": 105.133506774902
    },
    "Can Tho": {
        "latitude": 10.052888889,
        "longitude": 105.787138889
    },
    "Dai Ngai": {
        "latitude": 9.73400021,
        "longitude": 106.0739975
    },
    "Kontum": {
        "latitude": 14.3470802307129,
        "longitude": 108.034233093262
    },
    "Duc Xuyen": {
        "latitude": 12.2967700958252,
        "longitude": 107.975898742676
    },
    "Cau 14 (Buon Bur)": {
        "latitude": 12.61196816,
        "longitude": 107.9287695
    },
    "Ban Don": {
        "latitude": 12.897910118103,
        "longitude": 107.783126831055
    },
    "Cho Lach": {
        "latitude": 10.27212559,
        "longitude": 106.1260294
    },
    "Vam Nao": {
        "latitude": 10.5786504745483,
        "longitude": 105.363372802734
    },
    "Vam Kenh": {
        "latitude": 10.274299621582,
        "longitude": 106.73713684082
    },
    "Dinh An": {
        "latitude": 9.53083038330078,
        "longitude": 106.367782592773
    },
    "Tan Hiep": {
        "latitude": 10.11512777,
        "longitude": 105.2841379
    },
    "Vi Thanh": {
        "latitude": 9.780156116,
        "longitude": 105.4681185
    },
    "Phung Hiep": {
        "latitude": 9.810157865,
        "longitude": 105.8230733
    },
    "Cai Lay": {
        "latitude": 10.40911487,
        "longitude": 106.1210283
    },
    "Jinghong": {
        "latitude": 22.0159702301025,
        "longitude": 100.802223205566
    },
    "Giang Son": {
        "latitude": 12.5101699829102,
        "longitude": 108.183242797852
    }
}
