#!/usr/bin/python3
#
# SPDX-FileCopyrightText: ExtremeXT
# SPDX-License-Identifier: 	AGPL-3.0-only
#

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np


# Configurable values
LOW_NITS = 150  # Nits up to which it should be more accurate than usual.
LOW_WEIGHT = 5  # 1-10, "weight" of the lower nits.
MID_NITS = 500  # Middle nit value, HBM cutoff.
TARGET_POINTS = 10  # Number of points the map will have; this should be the lowest value possible that still generates a workable map.

AUTO_DECIMALS = 6


@dataclass
class Tuning:
    low_nits: float = LOW_NITS
    low_weight: float = LOW_WEIGHT
    mid_nits: float = MID_NITS
    target_points: int = TARGET_POINTS

    @classmethod
    def from_args(cls, args):
        return cls(
            low_nits=args.low_nits if args.low_nits is not None else LOW_NITS,
            low_weight=args.low_weight if args.low_weight is not None else LOW_WEIGHT,
            mid_nits=args.mid_nits if args.mid_nits is not None else MID_NITS,
            target_points=args.target_points
            if args.target_points is not None
            else TARGET_POINTS,
        )

    def validate(self):
        if self.target_points < 3:
            sys.exit("error: --target-points must be >= 3")


def _overlay_array(root, name):
    for arr in root.findall(".//integer-array"):
        if arr.get("name") == name:
            return [float(i.text.strip()) for i in arr.findall("item")]
    for arr in root.findall(".//array"):
        if arr.get("name") == name:
            return [float(i.text.strip()) for i in arr.findall("item")]
    raise KeyError(name)


def parse_overlay(path):
    root = ET.parse(path).getroot()

    backlight = _overlay_array(root, "config_screenBrightnessBacklight")
    nits = _overlay_array(root, "config_screenBrightnessNits")
    xs = np.array([b / max(backlight) for b in backlight])
    ys = np.array(nits, dtype=float)
    return xs, ys


def parse_auto_curve(path):
    # frameworks/base/core/res/res/values/config.xml#1737-1746
    # frameworks/base/services/core/java/com/android/server/display/DisplayDeviceConfig.java#3029-3037
    root = ET.parse(path).getroot()
    levels = _overlay_array(root, "config_autoBrightnessLevels")
    values = _overlay_array(root, "config_autoBrightnessDisplayValuesNits")
    if len(values) == len(levels) + 1:
        luxes = [0.0] + list(levels)
    elif len(values) == len(levels) and levels and levels[0] == 0:
        luxes = list(levels)
    else:
        raise ValueError(
            f"{path}: len(config_autoBrightnessDisplayValuesNits) must be "
            f"len(config_autoBrightnessLevels) + 1 "
            f"(got {len(values)} values, {len(levels)} levels)"
        )
    return [(float(lux), float(n)) for lux, n in zip(luxes, values)]


def generate_map_points(xs, ys, tuning):
    from scipy.interpolate import PchipInterpolator

    full = PchipInterpolator(xs, ys)
    inv = PchipInterpolator(ys, xs)

    selected_x = [0.0, float(inv(tuning.mid_nits)), 1.0]
    selected_y = [float(full(0.0)), float(tuning.mid_nits), float(full(1.0))]

    weight = np.where(ys < tuning.low_nits, tuning.low_weight, 1.0)

    while len(selected_x) < tuning.target_points:
        pred = PchipInterpolator(selected_x, selected_y)(xs)
        err_w = np.abs(pred - ys) * weight

        idx = int(np.argmax(err_w))
        wx, wy = float(xs[idx]), float(ys[idx])

        if wx in selected_x:
            for cand in np.argsort(err_w)[::-1]:
                if float(xs[cand]) not in selected_x:
                    wx, wy = float(xs[cand]), float(ys[cand])
                    break
            else:
                break

        selected_x.append(wx)
        selected_y.append(wy)
        selected_x, selected_y = map(list, zip(*sorted(zip(selected_x, selected_y))))

    return selected_x, selected_y


def make_interpolator(values, nits):
    # https://android.googlesource.com/platform/frameworks/base/+/refs/tags/android-17.0.0_r1/core/java/android/util/Spline.java#40
    # https://android.googlesource.com/platform/frameworks/base/+/refs/tags/android-17.0.0_r1/services/core/java/com/android/server/display/DisplayDeviceConfig.java#2737
    from scipy.interpolate import PchipInterpolator

    return PchipInterpolator(values, nits)


_spline_cache = {}


def _forward_spline(map_pts):
    # https://android.googlesource.com/platform/frameworks/base/+/refs/tags/android-17.0.0_r1/core/java/android/util/Spline.java#183
    from scipy.interpolate import PchipInterpolator

    key = tuple(map_pts)
    fwd = _spline_cache.get(key)
    if fwd is None:
        b = np.array([pt[0] for pt in map_pts], dtype=float)
        n = np.array([pt[1] for pt in map_pts], dtype=float)
        fwd = PchipInterpolator(b, n)
        _spline_cache[key] = fwd
    return fwd


def backlight_to_nits(map_pts, backlight):
    fwd = _forward_spline(map_pts)
    if backlight <= map_pts[0][0]:
        return map_pts[0][1]
    if backlight >= map_pts[-1][0]:
        return map_pts[-1][1]
    return float(fwd(backlight))


def nits_to_backlight(map_pts, nits):
    # https://android.googlesource.com/platform/frameworks/base/+/refs/tags/android-17.0.0_r1/services/core/java/com/android/server/display/DisplayDeviceConfig.java#2745
    # https://android.googlesource.com/platform/frameworks/base/+/refs/tags/android-17.0.0_r1/services/core/java/com/android/server/display/BrightnessMappingStrategy.java#949
    fwd = _forward_spline(map_pts)
    if nits <= map_pts[0][1]:
        return map_pts[0][0]
    if nits >= map_pts[-1][1]:
        return map_pts[-1][0]
    from scipy.optimize import brentq

    return float(
        brentq(
            lambda b: float(fwd(b)) - nits, map_pts[0][0], map_pts[-1][0], xtol=1e-12
        )
    )


def fmt(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s + ".0" if "." not in s else s


def fmt_nits(nits):
    s = f"{nits:.1f}"
    return s[:-2] if s.endswith(".0") else s


def print_screen_map(raw):
    print("<displayConfiguration>")
    print("    <screenBrightnessMap>")
    for value, nit in raw:
        print("        <point>")
        print(f"            <value>{value}</value>")
        print(f"            <nits>{nit}</nits>")
        print("        </point>")
    print("    </screenBrightnessMap>")
    print("</displayConfiguration>")


def render_lux_map(curve, map_pts, decimals):
    lines = ["<map>"]
    for lux, nits in curve:
        second = nits_to_backlight(map_pts, nits)
        rounded = round(second, decimals)
        actual = backlight_to_nits(map_pts, rounded)
        lines.append("    <point>")
        lines.append(f"        <first>{lux:g}</first>")
        lines.append(
            f"        <second>{rounded:.{decimals}f}</second>"
            f" <!-- {fmt_nits(actual)} nits -->"
        )
        lines.append("    </point>")
    lines.append("</map>")
    return lines


def render_auto_block(curve, map_pts, decimals):
    # https://android.googlesource.com/platform/frameworks/base/+/refs/tags/android-17.0.0_r1/services/core/java/com/android/server/display/config/DisplayBrightnessMappingConfig.java#100
    inner = render_lux_map(curve, map_pts, decimals)
    lines = [
        "<displayConfiguration>",
        '    <autoBrightness enabled="true">',
        "        <luxToBrightnessMapping>",
        "            <mode>default</mode>",
        "            <setting>normal</setting>",
    ]
    lines += ["            " + line for line in inner]
    lines.append("        </luxToBrightnessMapping>")
    lines.append("    </autoBrightness>")
    lines.append("</displayConfiguration>")
    return lines


def dark_axes(ax, title):
    ax.set_facecolor("black")
    ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(True, color="white", alpha=0.2)
    legend = ax.legend(facecolor="black", edgecolor="white")
    for text in legend.get_texts():
        text.set_color("white")


def screen_map(values, nits):
    import matplotlib.pyplot as plt

    interp = make_interpolator(values, nits)
    x = np.linspace(values[0], values[-1], 1000)
    y = interp(x)
    linear = np.interp(x, [values[0], values[-1]], [nits[0], nits[-1]])

    _, ax = plt.subplots(figsize=(8, 6), facecolor="black")
    ax.plot(x, y, color="white", linewidth=2, label="Nits (cubic)")
    ax.plot(x, linear, color="white", linestyle="--", linewidth=1.5, label="Linear")
    ax.scatter(values, nits, color="white", s=20)
    ax.set_xlabel("Relative backlight value", color="white")
    ax.set_ylabel("Nits", color="white")
    dark_axes(ax, "Brightness Curve")
    plt.tight_layout()
    plt.show()


def add_tuning_args(parser):
    parser.add_argument(
        "--low-nits",
        type=float,
        default=None,
        help=f"override LOW_NITS (default: {LOW_NITS})",
    )
    parser.add_argument(
        "--low-weight",
        type=float,
        default=None,
        help=f"override LOW_WEIGHT (default: {LOW_WEIGHT})",
    )
    parser.add_argument(
        "--mid-nits",
        type=float,
        default=None,
        help=f"override MID_NITS (default: {MID_NITS})",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=None,
        help=f"override TARGET_POINTS (default: {TARGET_POINTS})",
    )
    parser.add_argument(
        "--print-vars",
        action="store_true",
        help="print effective tuning variables to stderr and continue",
    )


def print_tuning(tuning, file=sys.stderr):
    for field in dataclass_fields(tuning):
        name = field.name.upper()
        value = getattr(tuning, field.name)
        default = globals()[name]
        suffix = "" if value == default else f" (default {default})"
        print(f"{name} = {value}{suffix}", file=file)


def build_map(overlay, tuning):
    if not Path(overlay).is_file():
        sys.exit(f"error: overlay not found: {overlay}")
    tuning.validate()
    xs, ys = parse_overlay(overlay)
    cx, cy = generate_map_points(xs, ys, tuning)
    return np.array(cx), np.array(cy)


def cmd_map(args):
    tuning = Tuning.from_args(args)
    values, nits = build_map(args.overlay, tuning)

    if args.curve:
        screen_map(values, nits)
    else:
        raw = [
            (fmt(x), f"{y:.1f}".rstrip("0").rstrip(".")) for x, y in zip(values, nits)
        ]
        print_screen_map(raw)


def cmd_auto(args):
    tuning = Tuning.from_args(args)
    values, nits = build_map(args.overlay, tuning)
    try:
        curve = parse_auto_curve(args.overlay)
    except (KeyError, ValueError, ET.ParseError) as err:
        sys.exit(f"error: {err}")
    map_pts = list(zip(values.tolist(), nits.tolist()))
    print("\n".join(render_auto_block(curve, map_pts, AUTO_DECIMALS)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True, metavar="{map,autobrightness}")

    p_map = sub.add_parser("map", help="print <screenBrightnessMap>")
    p_map.add_argument(
        "overlay",
        nargs="?",
        default="overlay.xml",
    )
    p_map.add_argument(
        "--curve",
        action="store_true",
    )
    add_tuning_args(p_map)
    p_map.set_defaults(func=cmd_map)

    p_auto = sub.add_parser("autobrightness", help="print <luxToBrightnessMapping>")
    p_auto.add_argument(
        "overlay",
        nargs="?",
        default="overlay.xml",
    )
    add_tuning_args(p_auto)
    p_auto.set_defaults(func=cmd_auto)

    args = ap.parse_args()
    if getattr(args, "print_vars", False):
        print_tuning(Tuning.from_args(args))
        return
    args.func(args)


if __name__ == "__main__":
    main()
