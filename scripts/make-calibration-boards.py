#!/usr/bin/env python3
"""Generate the calibration checkerboards (9x6 inner corners, 10x7 squares).

phone-nothing-2.png: 1080x2412 portrait, pixel-exact for the Nothing Phone (2)
  (6.7" 1080x2412 OLED, ~394 ppi). RGB intrinsics only: OLED emits no near-IR.
a4-laser.pdf: A4 landscape, 25 mm squares. Print at 100 % on a LASER printer;
  carbon toner is dark in near-IR, so the IR camera and stereo can use it.
"""
import struct
import sys
import zlib
from pathlib import Path

COLS, ROWS = 10, 7


def png_gray(path, width, height, pixel):
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(pixel(x, y) for x in range(width))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def phone_board(path):
    width, height, square = 1080, 2412, 134
    across, down = ROWS, COLS
    left = (width - across * square) // 2
    top = (height - down * square) // 2

    def pixel(x, y):
        i, j = x - left, y - top
        if not (0 <= i < across * square and 0 <= j < down * square):
            return 255
        return 0 if (i // square + j // square) % 2 == 0 else 255

    png_gray(path, width, height, pixel)
    ppi = (1080**2 + 2412**2) ** 0.5 / 6.7
    return square, square * 25.4 / ppi


def a4_board(path, square_mm=25.0):
    pt = 72 / 25.4
    page_w, page_h = 297 * pt, 210 * pt
    s = square_mm * pt
    x0 = (page_w - COLS * s) / 2
    y0 = (page_h - ROWS * s) / 2
    ops = ["0 g"]
    for j in range(ROWS):
        for i in range(COLS):
            if (i + j) % 2 == 0:
                y = page_h - y0 - (j + 1) * s
                ops.append(f"{x0 + i * s:.4f} {y:.4f} {s:.4f} {s:.4f} re f")
    note = "9x6 inner corners, 25.0 mm squares. Print at 100% (actual size) on a laser printer."
    ops.append(f"BT /F1 9 Tf {x0:.2f} {page_h - y0 + 12:.2f} Td ({note}) Tj ET")
    content = "\n".join(ops).encode()

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.4f} {page_h:.4f}] "
        f"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".encode(),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/calibration")
    out.mkdir(parents=True, exist_ok=True)
    square_px, square_mm = phone_board(out / "phone-nothing-2.png")
    a4_board(out / "a4-laser.pdf")
    print(f"phone: {square_px} px squares, ~{square_mm:.2f} mm at 394 ppi")
    print("a4: 25.0 mm squares")


if __name__ == "__main__":
    main()
