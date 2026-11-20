"""Sample the four B70 sysfs clocks and limits during a benchmark window."""
from pathlib import Path
from time import sleep, time_ns


def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def main():
    cards = []
    for bdf in ("0a", "0e", "14", "18"):
        slot = f"0000:{bdf}:00.0"
        root = Path("/sys/bus/pci/devices") / slot
        freq = root / "tile0/gt0/freq0"
        hwmon = next(path.parent for path in root.glob("hwmon/*/power1_cap"))
        cards.append((slot, [freq / file for file in ("min_freq", "max_freq", "cur_freq", "act_freq")]
                      + [hwmon / "power1_cap", hwmon / "temp1_input"]))
    print("epoch_ns\tpci_bdf\tmin_mhz\tmax_mhz\tcur_mhz\tact_mhz\tpower_cap_uw\ttemp_mc")
    while True:
        for slot, paths in cards:
            print("\t".join([str(time_ns()), slot, *(read(path) for path in paths)]))
        sleep(0.25)


if __name__ == "__main__":
    main()
