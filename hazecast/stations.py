"""DOE continuous air-quality monitoring stations used by APIMS.

Coordinates are approximate town-centre locations (to ~0.05 deg), used only for
the map and for picking each station's nearest neighbours. They are not the
exact sensor positions.
"""

# name: (lat, lon, state, region)
STATIONS = {
    "Alor Gajah": (2.38, 102.21, "Melaka", "Peninsular South"),
    "Alor Setar": (6.12, 100.37, "Kedah", "Peninsular North"),
    "Balik Pulau": (5.35, 100.24, "Pulau Pinang", "Peninsular North"),
    "Balok Baru Kuantan": (3.96, 103.38, "Pahang", "Peninsular East"),
    "Bandaraya Melaka": (2.19, 102.25, "Melaka", "Peninsular South"),
    "Banting": (2.82, 101.50, "Selangor", "Klang Valley"),
    "Batu Muda": (3.21, 101.68, "Kuala Lumpur", "Klang Valley"),
    "Batu Pahat": (1.85, 102.93, "Johor", "Peninsular South"),
    "Besut": (5.77, 102.56, "Terengganu", "Peninsular East"),
    "Bintulu": (3.17, 113.04, "Sarawak", "Sarawak"),
    "Bukit Rambai": (2.26, 102.17, "Melaka", "Peninsular South"),
    "Cheras": (3.10, 101.74, "Kuala Lumpur", "Klang Valley"),
    "ILP Miri": (4.42, 114.01, "Sarawak", "Sarawak"),
    "Indera Mahkota Kuantan": (3.82, 103.30, "Pahang", "Peninsular East"),
    "Jerantut": (3.94, 102.36, "Pahang", "Peninsular East"),
    "Kangar": (6.44, 100.20, "Perlis", "Peninsular North"),
    "Kapit": (2.01, 112.93, "Sarawak", "Sarawak"),
    "Kemaman": (4.23, 103.42, "Terengganu", "Peninsular East"),
    "Keningau": (5.34, 116.16, "Sabah", "Sabah & Labuan"),
    "Kimanis": (5.52, 115.86, "Sabah", "Sabah & Labuan"),
    "Klang": (3.04, 101.45, "Selangor", "Klang Valley"),
    "Kluang": (2.03, 103.32, "Johor", "Peninsular South"),
    "Kota Bharu": (6.13, 102.24, "Kelantan", "Peninsular East"),
    "Kota Kinabalu": (5.98, 116.07, "Sabah", "Sabah & Labuan"),
    "Kota Tinggi": (1.73, 103.90, "Johor", "Peninsular South"),
    "Kuala Selangor": (3.34, 101.25, "Selangor", "Klang Valley"),
    "Kuala Terengganu": (5.33, 103.14, "Terengganu", "Peninsular East"),
    "Kuching": (1.55, 110.36, "Sarawak", "Sarawak"),
    "Kulim": (5.41, 100.56, "Kedah", "Peninsular North"),
    "Labuan": (5.28, 115.24, "Labuan", "Sabah & Labuan"),
    "Langkawi": (6.35, 99.80, "Kedah", "Peninsular North"),
    "Larkin": (1.50, 103.74, "Johor", "Peninsular South"),
    "Limbang": (4.75, 115.01, "Sarawak", "Sarawak"),
    "Minden": (5.36, 100.30, "Pulau Pinang", "Peninsular North"),
    "Miri": (4.40, 113.99, "Sarawak", "Sarawak"),
    "Muar": (2.04, 102.57, "Johor", "Peninsular South"),
    "Mukah": (2.90, 112.09, "Sarawak", "Sarawak"),
    "Nilai": (2.82, 101.80, "Negeri Sembilan", "Klang Valley"),
    "Paka": (4.64, 103.44, "Terengganu", "Peninsular East"),
    "Pasir Gudang": (1.47, 103.90, "Johor", "Peninsular South"),
    "Pegoh Ipoh": (4.55, 101.08, "Perak", "Peninsular North"),
    "Pengerang": (1.37, 104.11, "Johor", "Peninsular South"),
    "Petaling Jaya": (3.11, 101.61, "Selangor", "Klang Valley"),
    "Port Dickson": (2.52, 101.80, "Negeri Sembilan", "Klang Valley"),
    "Presint 18": (2.91, 101.67, "Putrajaya", "Klang Valley"),
    "Putrajaya": (2.93, 101.69, "Putrajaya", "Klang Valley"),
    "Rompin": (2.80, 103.48, "Pahang", "Peninsular East"),
    "Samalaju": (3.53, 113.31, "Sarawak", "Sarawak"),
    "Samarahan": (1.46, 110.49, "Sarawak", "Sarawak"),
    "Sandakan": (5.84, 118.12, "Sabah", "Sabah & Labuan"),
    "Sarikei": (2.13, 111.52, "Sarawak", "Sarawak"),
    "Seberang Jaya": (5.40, 100.40, "Pulau Pinang", "Peninsular North"),
    "Seberang Perai": (5.38, 100.39, "Pulau Pinang", "Peninsular North"),
    "Segamat": (2.51, 102.82, "Johor", "Peninsular South"),
    "Seremban": (2.73, 101.94, "Negeri Sembilan", "Klang Valley"),
    "Seri Manjung": (4.21, 100.66, "Perak", "Peninsular North"),
    "Shah Alam": (3.07, 101.52, "Selangor", "Klang Valley"),
    "Sibu": (2.29, 111.83, "Sarawak", "Sarawak"),
    "Sri Aman": (1.24, 111.46, "Sarawak", "Sarawak"),
    "Sungai Petani": (5.65, 100.49, "Kedah", "Peninsular North"),
    "Taiping": (4.85, 100.74, "Perak", "Peninsular North"),
    "Tanah Merah": (5.80, 102.15, "Kelantan", "Peninsular East"),
    "Tangkak": (2.27, 102.54, "Johor", "Peninsular South"),
    "Tanjung Malim": (3.68, 101.52, "Perak", "Peninsular North"),
    "Tasek Ipoh": (4.63, 101.11, "Perak", "Peninsular North"),
    "Tawau": (4.24, 117.89, "Sabah", "Sabah & Labuan"),
    "Temerloh": (3.45, 102.42, "Pahang", "Peninsular East"),
}

REGIONS = sorted({v[3] for v in STATIONS.values()})


def nearest_neighbours(k: int = 3) -> dict:
    """Return {station: [k nearest other stations]} by great-circle distance."""
    import math

    def dist(a, b):
        la1, lo1 = map(math.radians, a)
        la2, lo2 = map(math.radians, b)
        h = (math.sin((la2 - la1) / 2) ** 2
             + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
        return 2 * 6371 * math.asin(math.sqrt(h))

    out = {}
    for s, (la, lo, *_rest) in STATIONS.items():
        d = sorted(
            (dist((la, lo), (v[0], v[1])), t)
            for t, v in STATIONS.items() if t != s
        )
        out[s] = [t for _, t in d[:k]]
    return out
