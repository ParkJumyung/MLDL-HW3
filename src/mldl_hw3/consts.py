from enum import Enum


class Brand(Enum):
    """
    Enum of recognized brands. All the values are in title-case. Transform to title-case before comparing.
    """

    Audi = "Audi"
    Bentley = "Bentley"
    BMW = "Bmw"
    Chevrolet = "Chevrolet"
    Datsun = "Datsun"
    Fiat = "Fiat"
    Force = "Force"
    Ford = "Ford"
    Honda = "Honda"
    Hyundai = "Hyundai"
    Isuzu = "Isuzu"
    Jaguar = "Jaguar"
    Jeep = "Jeep"
    Lamborghini = "Lamborghini"
    Land_Rover = "Land Rover"
    Mahindra = "Mahindra"
    Maruti = "Maruti"
    Mercedes_Benz = "Mercedes-Benz"
    Mini = "Mini"
    Mitsubishi = "Mitsubishi"
    Nissan = "Nissan"
    Porsche = "Porsche"
    Renault = "Renault"
    Skoda = "Skoda"
    Smart = "Smart"
    Tata = "Tata"
    Toyota = "Toyota"
    Volkswagen = "Volkswagen"
    Volvo = "Volvo"


fuel_densities = {"CNG": 1.33, "Diesel": 1.20, "LPG": 1.85, "Petrol": 1.35}

owner_type_order = ["First", "Second", "Third", "Fourth & Above"]
