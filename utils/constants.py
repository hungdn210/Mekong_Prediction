class Constants:
    def __init__(self):
        # constants
        self._water_flow_dict = {
            "Ban Chot": [
                "Ban Tad Ton",
                "Ban Nong Kiang"
            ],
            "Ban Huai Khayuong": [],
            "Ban Huai Yano Mai": [],
            "Ban Kengdone": [],
            "Ban Na Luang": [],
            "Ban Nong Kiang": [],
            "Ban Pak Huai": [],
            "Ban Pak Kanhoung": [
                "Ban Na Luang"
            ],
            "Ban Tad Ton": [],
            "Ban Tha Mai Liam": [],
            "Ban Tha Ton": [],
            "Cau 14 (Buon Bur)": [
                "Duc Xuyen"
            ],
            "Chaktomuk": [],
            "Chiang Khan": [
                "Vientiane KM4"
            ],
            "Chiang Saen": [
                "Ban Huai Yano Mai",
                "Ban Tha Ton",
                "Ban Tha Mai Liam",
                "Ban Pak Huai",
                "Chiang Khan"
            ],
            "Duc Xuyen": [],
            "Khong Chiam": [
                "Yasothon",
                "Pakse"
            ],
            "Kompong Cham": [
                "Chaktomuk"
            ],
            "Kontum": [],
            "Mukdahan": [
                "Ban Kengdone",
                "Khong Chiam"
            ],
            "Nakhon Phanom": [
                "Mukdahan"
            ],
            "Nong Khai": [
                "Ban Pak Kanhoung",
                "Nakhon Phanom"
            ],
            "Pakse": [
                "Ban Huai Khayuong",
                "Stung Treng"
            ],
            "Stung Treng": [
                "Kontum",
                "Cau 14 (Buon Bur)",
                "Kompong Cham"
            ],
            "Vientiane KM4": [
                "Nong Khai"
            ],
            "Yasothon": [
                "Ban Chot"
            ]
        }

        # self._all_stations = (
        #     'Chroy Chang Var', 'Kompong Cham', 'Chaktomuk', 'Chantangoy', 'Ban Kamphun', 'Battambang', 'Vientiane KM4',
        #     'Ban Pak Kanhoung', 'Ban Na Luang', 'Chiang Saen', 'Ban Huai Yano Mai', 'Ban Tha Ton', 'Ban Tha Mai Liam',
        #     'Ban Pak Huai', 'Yasothom', 'Ban Chot', 'Ban Nong Kiang', 'Ban Tad Ton', 'Ban Huai Khayuong',
        #     'Cau 14 (Buon Bur)', 'Stung Treng', 'Kratie', 'Lumphat', 'Pakse', 'Ban Kengdone', 'Chiang Khan',
        #     'Nong Khai', 'Nakhon Phanom', 'Mukdahan', 'Khong Chiam', 'Kontum', 'Duc Xuyen', 'Ban Don'
        # )
        self._all_stations = [
            'Kompong Cham', 'Chaktomuk', 'Vientiane KM4', 'Ban Pak Kanhoung',
            'Ban Na Luang', 'Chiang Saen', 'Ban Huai Yano Mai', 'Ban Tha Ton',
            'Ban Tha Mai Liam', 'Ban Pak Huai', 'Yasothon', 'Ban Chot',
            'Ban Nong Kiang', 'Ban Tad Ton', 'Ban Huai Khayuong',
            'Cau 14 (Buon Bur)', 'Stung Treng', 'Pakse', 'Ban Kengdone',
            'Chiang Khan', 'Nong Khai', 'Nakhon Phanom', 'Mukdahan',
            'Khong Chiam', 'Kontum', 'Duc Xuyen'
        ]

        self._cross_list = []
        for station1 in self._water_flow_dict.keys():
            for station2 in self._water_flow_dict[station1]:
                self._cross_list.append((station1, station2))

        self._training_range = '1992-09-01 00:00:00+00:00 to 2016-08-31 00:00:00+00:00'
        self._validation_range = '2016-09-01 00:00:00+00:00 to 2020-08-31 00:00:00+00:00'
        self._testing_range = '2020-09-01 00:00:00+00:00 to 2023-09-31 00:00:00+00:00'

    @property
    def water_flow_dict(self):
        return self._water_flow_dict

    @property
    def all_stations(self):
        return self._all_stations

    @property
    def cross_list(self):
        return self._cross_list

    @property
    def training_range(self):
        return self._training_range

    @property
    def validation_range(self):
        return self._validation_range

    @property
    def testing_range(self):
        return self._testing_range
