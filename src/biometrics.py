# src/biometrics.py

def estimate_dimensions(box, reference_length_cm=None, reference_px=None):
    """
    box: (x1, y1, x2, y2) from YOLO detection
    If reference object is provided, computes real-world size.
    Otherwise returns pixel dimensions only.
    """
    x1, y1, x2, y2 = box
    pixel_length = x2 - x1   # horizontal span
    pixel_width  = y2 - y1   # vertical span

    if reference_length_cm and reference_px:
        scale = reference_length_cm / reference_px   # cm per pixel
        real_length = round(pixel_length * scale, 1)
        real_width  = round(pixel_width  * scale, 1)
        return real_length, real_width
    else:
        # Return pixel values with disclaimer
        return pixel_length, pixel_width

# src/biometrics.py  (add below estimate_dimensions)

WEIGHT_COEFFICIENTS = {
    "GoldFish":               {"a": 0.0141, "b": 3.02},
    "ClownFish":              {"a": 0.0120, "b": 3.10},
    "ZebraFish":              {"a": 0.0083, "b": 3.15},
    "AngelFish":              {"a": 0.0160, "b": 3.05},
    "BlueTang":               {"a": 0.0175, "b": 3.00},
    "ButterflyFish":          {"a": 0.0130, "b": 3.08},
    "Gourami":                {"a": 0.0110, "b": 3.12},
    "MorishIdol":             {"a": 0.0145, "b": 3.01},
    "PlatyFish":              {"a": 0.0095, "b": 3.18},
    "RibbonedSweetlips":      {"a": 0.0200, "b": 2.98},
    "ThreeStripedDamselfish": {"a": 0.0105, "b": 3.14},
    "YellowCichlid":          {"a": 0.0135, "b": 3.06},
    "YellowTang":             {"a": 0.0150, "b": 3.03},
}

MATURITY_LENGTH_CM = {
    "GoldFish": 10, "ClownFish": 8, "ZebraFish": 4,
    "AngelFish": 12, "BlueTang": 15, "ButterflyFish": 10,
    "Gourami": 7, "MorishIdol": 18, "PlatyFish": 4,
    "RibbonedSweetlips": 20, "ThreeStripedDamselfish": 7,
    "YellowCichlid": 10, "YellowTang": 15,
}

# Maximum biologically plausible length (cm) per species.
# Values include a generous 2× margin above real-world max to
# accommodate depth-measurement noise — anything beyond this is
# definitely NOT a fish (e.g. a 154 cm "ClownFish" = person).
MAX_LENGTH_CM = {
    "GoldFish":               60,    # real max ~30 cm (Comet type)
    "ClownFish":              25,    # real max ~12 cm
    "ZebraFish":              12,    # real max ~5 cm
    "AngelFish":              60,    # real max ~30 cm (freshwater)
    "BlueTang":               60,    # real max ~31 cm
    "ButterflyFish":          50,    # real max ~25 cm
    "Gourami":                50,    # real max ~25 cm (Giant: up to 70 cm)
    "MorishIdol":             50,    # real max ~23 cm
    "PlatyFish":              15,    # real max ~7 cm
    "RibbonedSweetlips":      80,    # real max ~40 cm
    "ThreeStripedDamselfish": 20,    # real max ~10 cm
    "YellowCichlid":          40,    # real max ~20 cm
    "YellowTang":             50,    # real max ~20 cm
}

def predict_weight(species, length_cm):
    """Returns predicted weight in grams using W = a * L^b"""
    coeffs = WEIGHT_COEFFICIENTS.get(species)
    if not coeffs or length_cm <= 0:
        return None
    weight = coeffs["a"] * (length_cm ** coeffs["b"])
    return round(weight, 1)

def classify_maturity(species, length_cm):
    maturity = MATURITY_LENGTH_CM.get(species, 10)
    return "Adult" if length_cm >= maturity else "Juvenile"