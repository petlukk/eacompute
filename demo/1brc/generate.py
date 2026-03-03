#!/usr/bin/env python3
"""Generate 1BRC test data: StationName;Temperature\n"""

import os
import random
import sys
import time

STATIONS = [
    "Abha", "Abidjan", "Abu Dhabi", "Accra", "Addis Ababa", "Adelaide",
    "Aden", "Ahmedabad", "Aleppo", "Alexandria", "Algiers", "Alice Springs",
    "Almaty", "Amsterdam", "Anadyr", "Anchorage", "Andorra la Vella", "Ankara",
    "Antananarivo", "Apia", "Ashgabat", "Asmara", "Astana", "Athens",
    "Atlanta", "Auckland", "Austin", "Baghdad", "Bahrain", "Baku",
    "Baltimore", "Bamako", "Bangkok", "Bangui", "Banjul", "Barcelona",
    "Beira", "Beirut", "Belgrade", "Belize City", "Benghazi", "Bergen",
    "Berlin", "Bern", "Bilbao", "Birmingham", "Bishkek", "Bissau",
    "Blantyre", "Bogota", "Bordeaux", "Bosaso", "Boston", "Bouake",
    "Bratislava", "Brazzaville", "Bridgetown", "Brisbane", "Brussels",
    "Bucharest", "Budapest", "Buenos Aires", "Bujumbura", "Bulawayo",
    "Bursa", "Busan", "Cabo San Lucas", "Cairns", "Cairo", "Calgary",
    "Canberra", "Cape Town", "Casablanca", "Changsha", "Charlotte",
    "Chengdu", "Chennai", "Chicago", "Chihuahua", "Chittagong", "Chongqing",
    "Christchurch", "City of San Marino", "Colombo", "Columbus", "Conakry",
    "Copenhagen", "Cotonou", "Cracow", "Da Nang", "Dakar", "Dallas",
    "Damascus", "Dar es Salaam", "Darwin", "Delhi", "Denpasar", "Denver",
    "Detroit", "Dhaka", "Dili", "Djibouti", "Dodoma", "Doha", "Douala",
    "Dubai", "Dublin", "Durban", "Dushanbe", "Edinburgh", "Edmonton",
    "El Paso", "Entebbe", "Erbil", "Faisalabad", "Florence", "Fortaleza",
    "Frankfurt", "Freetown", "Fukuoka", "Gaborone", "Gander", "Geneva",
    "Genoa", "Georgetown", "Glasgow", "Gothenburg", "Guadalajara",
    "Guangzhou", "Guatemala City", "Halifax", "Hamburg", "Hanoi", "Harare",
    "Harbin", "Havana", "Helsinki", "Hiroshima", "Ho Chi Minh City",
    "Hobart", "Hong Kong", "Honolulu", "Houston", "Hyderabad", "Ibadan",
    "Indianapolis", "Islamabad", "Istanbul", "Jacksonville", "Jakarta",
    "Jeddah", "Jerusalem", "Johannesburg", "Kabul", "Kampala", "Kano",
    "Kansas City", "Karachi", "Kathmandu", "Khartoum", "Kiev", "Kigali",
    "Kingston", "Kinshasa", "Kolkata", "Kuala Lumpur", "Kumasi", "Kunming",
    "Kuwait City", "Kyoto", "La Paz", "Lagos", "Lahore", "Las Palmas",
    "Las Vegas", "Leeds", "Leipzig", "Libreville", "Lima", "Lisbon",
    "Ljubljana", "Lome", "London", "Los Angeles", "Louisville", "Luanda",
    "Lubumbashi", "Lusaka", "Luxembourg City", "Lviv", "Lyon", "Madrid",
    "Makassar", "Malabo", "Male", "Managua", "Manama", "Mandalay",
    "Manila", "Maputo", "Marrakesh", "Marseille", "Mecca", "Medan",
    "Medina", "Melbourne", "Memphis", "Mexicali", "Mexico City", "Miami",
    "Milan", "Milwaukee", "Minneapolis", "Minsk", "Mogadishu", "Monaco",
    "Moncton", "Monterrey", "Montreal", "Moscow", "Mumbai", "Munich",
    "Murmansk", "Muscat", "Mwanza", "Nairobi", "Nanning", "Naples",
    "Nashville", "Nassau", "Ndjamena", "New Orleans", "New York City",
    "Niamey", "Nice", "Nicosia", "Nouakchott", "Novosibirsk", "Nuuk",
    "Odessa", "Okayama", "Oklahoma City", "Omaha", "Oranjestad", "Orlando",
    "Osaka", "Oslo", "Ottawa", "Ouagadougou", "Palermo", "Palm Springs",
    "Palma de Mallorca", "Panama City", "Paramaribo", "Paris", "Patna",
    "Perth", "Philadelphia", "Phnom Penh", "Phoenix", "Pittsburgh",
    "Podgorica", "Pontianak", "Port Elizabeth", "Port Moresby",
    "Port of Spain", "Portland", "Porto", "Prague", "Pretoria",
    "Pyongyang", "Quebec City", "Quito", "Rabat", "Raleigh", "Rangoon",
    "Recife", "Reykjavik", "Riga", "Rio de Janeiro", "Riyadh", "Rome",
    "Roseau", "Rostov-on-Don", "Sacramento", "Saint Petersburg",
    "Salt Lake City", "San Antonio", "San Diego", "San Francisco",
    "San Jose", "San Juan", "San Salvador", "Sanaa", "Santiago",
    "Santo Domingo", "Sao Paulo", "Sapporo", "Sarajevo", "Seattle",
    "Seoul", "Seville", "Shanghai", "Singapore", "Skopje", "Sofia",
    "St. Louis", "Stockholm", "Surabaya", "Surat", "Suva", "Sydney",
    "Tabriz", "Taipei", "Tallinn", "Tampa", "Tashkent", "Tbilisi",
    "Tegucigalpa", "Tehran", "Tel Aviv", "Thessaloniki", "Thimphu",
    "Tirana", "Tokyo", "Toronto", "Tripoli", "Tucson", "Tunis", "Turin",
    "Ulaanbaatar", "Valencia", "Valletta", "Vancouver", "Victoria",
    "Vienna", "Vientiane", "Vilnius", "Warsaw", "Washington", "Wellington",
    "Whitehorse", "Wuhan", "Xian", "Yakutsk", "Yangon", "Yaounde",
    "Yerevan", "Yokohama", "Zagreb", "Zanzibar City", "Zurich",
]

BUFFER_SIZE = 65536


def generate(n_rows: int, output_path: str) -> None:
    rng = random.Random(42)
    n_stations = len(STATIONS)
    start = time.time()
    written = 0

    with open(output_path, "wb") as f:
        buf = bytearray()
        for row in range(n_rows):
            station = STATIONS[rng.randint(0, n_stations - 1)]
            # Temperature: uniform in [-20.0, 45.0], one decimal
            temp_tenths = rng.randint(-200, 450)
            sign = "-" if temp_tenths < 0 else ""
            abs_val = abs(temp_tenths)
            line = f"{station};{sign}{abs_val // 10}.{abs_val % 10}\n"
            buf.extend(line.encode())
            if len(buf) >= BUFFER_SIZE:
                f.write(buf)
                written += len(buf)
                buf.clear()
        if buf:
            f.write(buf)
            written += len(buf)

    elapsed = time.time() - start
    mb = written / (1024 * 1024)
    print(f"Generated {n_rows:,} rows -> {output_path} ({mb:.1f} MB, {elapsed:.1f}s)")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <n_rows> [output_path]")
        print(f"  n_rows: number of rows to generate")
        print(f"  output_path: optional, default: data/measurements_<suffix>.txt")
        sys.exit(1)

    n_rows = int(sys.argv[1])

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        if n_rows >= 1_000_000_000:
            suffix = "1B"
        elif n_rows >= 1_000_000:
            suffix = f"{n_rows // 1_000_000}M"
        elif n_rows >= 1_000:
            suffix = f"{n_rows // 1_000}K"
        else:
            suffix = str(n_rows)
        output_path = os.path.join(data_dir, f"measurements_{suffix}.txt")

    generate(n_rows, output_path)


if __name__ == "__main__":
    main()
