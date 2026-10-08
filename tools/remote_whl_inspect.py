import io, re, sys, zipfile, urllib.request

URL = sys.argv[1]


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req, timeout=30).headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        data = urllib.request.urlopen(req, timeout=120).read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


zf = zipfile.ZipFile(io.BufferedReader(HttpFile(URL), buffer_size=1 << 20))
names = zf.namelist()
meta = [n for n in names if n.endswith(".dist-info/METADATA")][0]
for line in zf.read(meta).decode().splitlines():
    if re.match(r"^(Version|Requires-Dist: (torch|flashinfer|nvidia|cuda))", line):
        print(line)
sos = [n for n in names if n.endswith(".so") and "/" not in n.split("vllm/", 1)[-1]]
print("top-level .so:", sos)
target = [n for n in names if "_C_stable_libtorch" in n]
print("stable ext:", target, [zf.getinfo(n).file_size for n in target])
if target and zf.getinfo(target[0]).file_size < 400 * 1024 * 1024:
    blob = zf.read(target[0])
    print("cuda libs referenced:", sorted(set(re.findall(rb"lib(?:cudart|cublas|cuda)\.so\.\d+", blob))))
