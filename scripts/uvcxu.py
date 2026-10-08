import ctypes, fcntl, os, sys
UVCIOC_CTRL_QUERY = 0xC0107521
Q = dict(SET_CUR=0x01, GET_CUR=0x81, GET_MIN=0x82, GET_MAX=0x83, GET_RES=0x84, GET_LEN=0x85, GET_INFO=0x86, GET_DEF=0x87)
class XU(ctypes.Structure):
    _fields_ = [("unit", ctypes.c_uint8), ("selector", ctypes.c_uint8), ("query", ctypes.c_uint8),
                ("size", ctypes.c_uint16), ("data", ctypes.POINTER(ctypes.c_uint8))]
def query(fd, unit, sel, q, size, payload=None):
    buf = (ctypes.c_uint8 * size)(*(payload or []))
    x = XU(unit, sel, Q[q], size, ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)))
    fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, x)
    return bytes(buf)
def dump(dev, unit, sel):
    fd = os.open(dev, os.O_RDWR)
    try:
        ln = int.from_bytes(query(fd, unit, sel, "GET_LEN", 2), "little")
        info = query(fd, unit, sel, "GET_INFO", 1)[0]
        out = [f"{dev} unit={unit} sel={sel} len={ln} info={info:#04x}"]
        for q in ("GET_CUR", "GET_MIN", "GET_MAX", "GET_RES", "GET_DEF"):
            try: out.append(f"  {q}: {query(fd, unit, sel, q, ln).hex(' ')}")
            except OSError as e: out.append(f"  {q}: err {e.errno}")
        print("\n".join(out))
    except OSError as e:
        print(f"{dev} unit={unit} sel={sel}: err {e.errno} {e.strerror}")
    finally: os.close(fd)
if __name__ == "__main__":
    if sys.argv[1] == "set":
        dev, unit, sel = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        payload = bytes.fromhex(sys.argv[5])
        fd = os.open(dev, os.O_RDWR); query(fd, unit, sel, "SET_CUR", len(payload), list(payload)); os.close(fd)
        print("set ok"); dump(dev, unit, sel)
    else:
        for spec in sys.argv[1:]:
            dev, unit, sel = spec.split(":"); dump(dev, int(unit), int(sel))
