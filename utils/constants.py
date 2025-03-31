class Constants:
    def __init__(self):
        # constants
        self._water_flow_dict = {
            "Stung Treng": [
                "Kratie"
            ],
            "Kratie": [
                "Kompong Cham"
            ],
            "Kompong Cham": [],
            "Phnom Penh Port": [],
            "Chaktomuk": [
                "Phnom Penh Port"
            ],
            "Lumphat": [
                "Ban Don"
            ],
            "Luang Prabang": [
                "Chiang Khan"
            ],
            "Vientiane KM4": [
                "Nong Khai"
            ],
            "Pakse": [
                "Stung Treng"
            ],
            "Ban Pak Kanhoung": [],
            "Ban Kengdone": [],
            "Chiang Saen": [
                "Luang Prabang"
            ],
            "Chiang Khan": [
                "Vientiane KM4"
            ],
            "Nong Khai": [
                "Ban Pak Kanhoung",
                "Nakhon Phanom"
            ],
            "Nakhon Phanom": [
                "Mukdahan"
            ],
            "Mukdahan": [
                "Ban Kengdone",
                "Khong Chiam"
            ],
            "Khong Chiam": [
                "Pakse"
            ],
            "Tan Chau": [
                "Chau Doc",
                "Vam Nao",
                "My Thuan"
            ],
            "My Thuan": [],
            "Chau Doc": [
                "Can Tho"
            ],
            "Can Tho": [],
            "Kontum": [],
            "Duc Xuyen": [],
            "Ban Don": [],
            "Vam Nao": [
                "Can Tho"
            ],
            "Vam Kenh": [],
            "Phung Hiep": []
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

        self._training_range = '1989-01-01 00:00:00+00:00 to 1997-12-31 00:00:00+00:00'
        self._validation_range = '1998-01-01 00:00:00+00:00 to 1999-12-31 00:00:00+00:00'
        self._testing_range = '2000-09-01 00:00:00+00:00 to 2002-10-31 00:00:00+00:00'

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
